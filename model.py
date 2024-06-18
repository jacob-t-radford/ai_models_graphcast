# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import dataclasses
import datetime
import functools
import gc
import logging
import os
from functools import cached_property
import climetlab as cml
from netCDF4 import Dataset as DS
import xarray
import numpy as np

from ai_models.model import Model

from .input import create_training_xarray,create_training_xarray_nc
from .output import save_output_xarray

LOG = logging.getLogger(__name__)


try:
    import haiku as hk
    import jax
    from graphcast import (
        autoregressive,
        casting,
        checkpoint,
        data_utils,
        graphcast,
        normalization,
    )
except ModuleNotFoundError as e:
    msg = "You need to install Graphcast from git to use this model. See README.md for details."
    LOG.error(msg)
    raise ModuleNotFoundError(f"{msg}\n{e}")


class GraphcastModel(Model):
    download_url = "https://storage.googleapis.com/dm_graphcast/{file}"
    expver = "dmgc"

    # Download
    download_files = [
        (
            "params/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 -"
            " pressure levels 13 - mesh 2to6 - precipitation output only.npz"
        ),
        "stats/diffs_stddev_by_level.nc",
        "stats/mean_by_level.nc",
        "stats/stddev_by_level.nc",
    ]

    # Input
    area = [90, 0, -90, 360]
    grid = [0.25, 0.25]

    param_sfc = [
        "lsm",
        "2t",
        "msl",
        "10u",
        "10v",
        "tp",
        "z",
    ]

    param_level_pl = (
        ["t", "z", "u", "v", "w", "q"],
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    )

    forcing_variables = [
        "toa_incident_solar_radiation",
        # Not calling julian day and day here, due to differing assumptions with Deepmind
        # Those forcings are created by graphcast.data_utils
    ]

    use_an = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hour_steps = 6
        self.lagged = [-6, 0]
        self.params = None
        self.ordering = self.param_sfc + [
            f"{param}{level}"
            for param in self.param_level_pl[0]
            for level in self.param_level_pl[1]
        ]

    # Jax doesn't seem to like passing configs as args through the jit. Passing it
    # in via partial (instead of capture by closure) forces jax to invalidate the
    # jit cache if you change configs.
    def _with_configs(self, fn):
        return functools.partial(
            fn,
            model_config=self.model_config,
            task_config=self.task_config,
        )

    # Always pass params and state, so the usage below are simpler
    def _with_params(self, fn):
        return functools.partial(fn, params=self.params, state=self.state)

    # Deepmind models aren't stateful, so the state is always empty, so just return the
    # predictions. This is requiredy by the rollout code, and generally simpler.
    @staticmethod
    def _drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    def load_model(self):
        with self.timer(f"Loading {self.download_files[0]}"):

            def get_path(filename):
                return os.path.join(self.assets, filename)

            diffs_stddev_by_level = xarray.load_dataset(
                get_path(self.download_files[1])
            ).compute()

            mean_by_level = xarray.load_dataset(
                get_path(self.download_files[2])
            ).compute()

            stddev_by_level = xarray.load_dataset(
                get_path(self.download_files[3])
            ).compute()

            def construct_wrapped_graphcast(model_config, task_config):
                """Constructs and wraps the GraphCast Predictor."""
                # Deeper one-step predictor.
                predictor = graphcast.GraphCast(model_config, task_config)

                # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
                # from/to float32 to/from BFloat16.
                predictor = casting.Bfloat16Cast(predictor)

                # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
                # BFloat16 happens after applying normalization to the inputs/targets.
                predictor = normalization.InputsAndResiduals(
                    predictor,
                    diffs_stddev_by_level=diffs_stddev_by_level,
                    mean_by_level=mean_by_level,
                    stddev_by_level=stddev_by_level,
                )

                # Wraps everything so the one-step model can produce trajectories.
                predictor = autoregressive.Predictor(
                    predictor,
                    gradient_checkpointing=True,
                )
                return predictor

            @hk.transform_with_state
            def run_forward(
                model_config,
                task_config,
                inputs,
                targets_template,
                forcings,
            ):
                predictor = construct_wrapped_graphcast(model_config, task_config)
                return predictor(
                    inputs,
                    targets_template=targets_template,
                    forcings=forcings,
                )

            with open(get_path(self.download_files[0]), "rb") as f:
                self.ckpt = checkpoint.load(f, graphcast.CheckPoint)
                self.params = self.ckpt.params
                self.state = {}

                self.model_config = self.ckpt.model_config
                self.task_config = self.ckpt.task_config

                LOG.info("Model description: %s", self.ckpt.description)
                LOG.info("Model license: %s", self.ckpt.license)

            jax.jit(self._with_configs(run_forward.init))
            self.model = self._drop_state(
                self._with_params(jax.jit(self._with_configs(run_forward.apply)))
            )

    @cached_property
    def start_date(self) -> "datetime":
        return self.all_fields.order_by(valid_datetime="descending")[0].datetime

    def run(self):
        with self.timer("Building model"):
            self.load_model()
        # all_fields = self.all_fields.to_xarray()

        if not self.file:
            with self.timer("Creating input data (total)"):
                with self.timer("Creating training data"):
                    training_xarray, time_deltas = create_training_xarray(
                        fields_sfc=self.fields_sfc,
                        fields_pl=self.fields_pl,
                        lagged=self.lagged,
                        start_date=self.start_date,
                        hour_steps=self.hour_steps,
                        lead_time=self.lead_time,
                        forcing_variables=self.forcing_variables,
                        constants=self.override_constants,
                        timer=self.timer,
                    )

                gc.collect()
                if self.debug:
                    training_xarray.to_netcdf("training_xarray.nc")

                with self.timer("Extracting input targets"):
                    (
                        input_xr,
                        template,
                        forcings,
                    ) = data_utils.extract_inputs_targets_forcings(
                        training_xarray,
                        target_lead_times=[
                            f"{int(delta.days * 24 + delta.seconds/3600):d}h"
                            for delta in time_deltas[len(self.lagged) :]
                        ],
                        **dataclasses.asdict(self.task_config),
                    )

                if self.debug:
                    input_xr.to_netcdf("input_xr.nc")
                    forcings.to_netcdf("forcings_xr.nc")

        else:
            time_deltas = [
                datetime.timedelta(hours=h)
                for h in self.lagged
                + [hour for hour in range(self.hour_steps, self.lead_time + self.hour_steps, self.hour_steps)]
            ]
            training_xarray = xarray.open_dataset(self.file)
            (
                input_xr,
                template,
                forcings,
            ) = data_utils.extract_inputs_targets_forcings(
                training_xarray,
                target_lead_times=[
                    f"{int(delta.days * 24 + delta.seconds/3600):d}h"
                    for delta in time_deltas[len(self.lagged) :]
                ],
                **dataclasses.asdict(self.task_config),
            )
        #input_xr.to_netcdf("input_xr_%s%s.nc" % (str(self.date),str(self.time)))
        #forcings.to_netcdf("forcings_xr_%s%s.nc" % (str(self.date),str(self.time)))
        with self.timer("Doing full rollout prediction in JAX"):
            output = self.model(
                rng=jax.random.PRNGKey(0),
                inputs=input_xr,
                targets_template=template,
                forcings=forcings,
            )

            if self.debug:
                output.to_netcdf("output.nc")

        if 'g' in self.ncorgrib:
            with self.timer("Saving output data"):
                all_fields = cml.load_source("file","newsample.grib")
                save_output_xarray(
                    output=output,
                    write=self.write,
                    target_variables=self.task_config.target_variables,
                    all_fields=all_fields,
                    ordering=self.ordering,
                    lead_time=self.lead_time,
                    hour_steps=self.hour_steps,
                    lagged=self.lagged,
                    inputdata=input_xr,
                    initdate=self.date,
                    inittime=self.time
                )
        if 'n' in self.ncorgrib:
            out = {}
            out['u10'] = {'values':np.zeros((self.lead_time // self.hour_steps + 1, 721, 1440)),'name':'10 metre U wind component','units':'m s-1'}
            out['v10'] = {'values':np.zeros((self.lead_time // self.hour_steps + 1, 721, 1440)),'name':'10 metre V wind component','units':'m s-1'}
            out['t2'] = {'values':np.zeros((self.lead_time // self.hour_steps + 1, 721, 1440)),'name':'2 metre temperature','units':'K'}
            out['msl'] = {'values':np.zeros((self.lead_time // self.hour_steps + 1, 721, 1440)),'name':'Pressure reduced to MSL','units':'Pa'}
            out['apcp'] = {'values':np.zeros((self.lead_time // self.hour_steps + 1, 721, 1440)),'name':'6-hr accumulated precipitation','units':'m'}
            out['t'] = {'values':np.zeros((self.lead_time // self.hour_steps + 1, 13, 721, 1440)),'name':'Temperature','units':'K'}
            out['u'] = {'values':np.zeros((self.lead_time // self.hour_steps + 1, 13, 721, 1440)),'name':'U component of wind','units':'m s-1'}
            out['v'] = {'values':np.zeros((self.lead_time // self.hour_steps + 1, 13, 721, 1440)),'name':'V component of wind','units':'m s-1'}
            out['z'] = {'values':np.zeros((self.lead_time // self.hour_steps + 1, 13, 721, 1440)),'name':'Geopotential','units':'m2 s-2'}
            out['q'] = {'values':np.zeros((self.lead_time // self.hour_steps + 1, 13, 721, 1440)),'name':'Specific humidity','units':'g kg-1'}
            out['w'] = {'values':np.zeros((self.lead_time // self.hour_steps + 1, 13, 721, 1440)),'name':'Vertical velocity','units':'Pa s-1'}
            print('step -1')
            print(input_xr['10m_u_component_of_wind'][0,0])
            out['t']['values'][0,:,:,:] = input_xr['temperature'][0,1,::-1,::-1,:]
            out['u']['values'][0,:,:,:] = input_xr['u_component_of_wind'][0,1,::-1,::-1,:]
            out['v']['values'][0,:,:,:] = input_xr['v_component_of_wind'][0,1,::-1,::-1,:]
            out['z']['values'][0,:,:,:] = input_xr['geopotential'][0,1,::-1,::-1,:]
            out['q']['values'][0,:,:,:] = input_xr['specific_humidity'][0,1,::-1,::-1,:]
            out['w']['values'][0,:,:,:] = input_xr['vertical_velocity'][0,1,::-1,::-1,:]

            out['u10']['values'][0,:,:] = input_xr['10m_u_component_of_wind'][0,1,::-1,:]
            out['v10']['values'][0,:,:] = input_xr['10m_v_component_of_wind'][0,1,::-1,:]
            out['msl']['values'][0,:,:] = input_xr['mean_sea_level_pressure'][0,1,::-1,:]
            out['t2']['values'][0,:,:] =  input_xr['2m_temperature'][0,1,::-1,:]
            out['apcp']['values'][0,:,:] = np.zeros((721,1440))

            out['t']['values'][1:,:,:,:] = output['temperature'][:,0,::-1,::-1,:]
            out['u']['values'][1:,:,:,:] = output['u_component_of_wind'][:,0,::-1,::-1,:]
            out['v']['values'][1:,:,:,:] = output['v_component_of_wind'][:,0,::-1,::-1,:]
            out['z']['values'][1:,:,:,:] = output['geopotential'][:,0,::-1,::-1,:]
            out['q']['values'][1:,:,:,:] = output['specific_humidity'][:,0,::-1,::-1,:]
            out['w']['values'][1:,:,:,:] = output['vertical_velocity'][:,0,::-1,::-1,:]
            print('batch check')
            print(output['10m_u_component_of_wind'])
            print('step 1')
            print(output['10m_u_component_of_wind'][0])
            out['u10']['values'][1:,:,:] = output['10m_u_component_of_wind'][:,0,::-1,:]
            out['v10']['values'][1:,:,:] = output['10m_v_component_of_wind'][:,0,::-1,:]
            out['msl']['values'][1:,:,:] = output['mean_sea_level_pressure'][:,0,::-1,:]
            out['t2']['values'][1:,:,:] =  output['2m_temperature'][:,0,::-1,:]

            out['apcp']['values'][1:,:,:] = output['total_precipitation_6hr'][:,0,::-1,:]
#            apcpcopy = np.copy(out['apcp']['values'])
#            for precipstep in range(1,41):
#                out['apcp']['values'][precipstep,:,:] = apcpcopy[precipstep,:,:] - apcpcopy[precipstep-1,:,:]

            outdir = self.path + ".nc"
            print(outdir)
            f = DS(outdir, 'w', format='NETCDF4')
            f.createDimension('time', self.lead_time // self.hour_steps + 1)
            f.createDimension('level', 13)
            f.createDimension('longitude', 1440)
            f.createDimension('latitude', 721)
            year = str(self.date)[0:4]
            month = str(self.date)[4:6]
            day = str(self.date)[6:8]
            hh = str(self.time).zfill(4)[0:2]
            initdt = datetime.datetime.strptime(f"{year}{month}{day}{hh}","%Y%m%d%H")
            inityr = str(initdt.year)
            initmnth = str(initdt.month).zfill(2)
            initday = str(initdt.day).zfill(2)
            inithr = str(initdt.hour).zfill(2)
            times = []
            for i in np.arange(0,self.lead_time + self.hour_steps,self.hour_steps):
                times.append(int((initdt + datetime.timedelta(hours=int(i))).timestamp()))
            time = f.createVariable('time', 'i4', ('time',))
            time[:] = np.array(times)
            time.setncattr('long_name','Date and Time')
            time.setncattr('units','seconds since 1970-1-1')
            time.setncattr('calendar','standard')
            lon = f.createVariable('longitude', 'f4', ('longitude',))
            lon[:] = np.arange(0,360,0.25)
            lon.setncattr('long_name','Longitude')
            lon.setncattr('units','degree')
            lat = f.createVariable('latitude', 'f4', ('latitude',))
            lat[:] = np.arange(-90,90.25,0.25)[::-1]
            lat.setncattr('long_name','Latitude')
            lat.setncattr('units','degree')
            levels = f.createVariable('level', 'i4', ('level',))
            levels[:] = np.array([50,100,150,200,250,300,400,500,600,700,850,925,1000])[::-1]
            levels.setncattr('long_name','Isobaric surfaces')
            levels.setncattr('units','hPa')
            for variable in ['u10','v10','t2','msl','t','u','v','z','q','w','apcp']:
                if variable in ['u','v','z','t','q','w']:
                    myvar = f.createVariable(variable,'f4',('time','level','latitude','longitude'))
                elif variable in ['u10','v10','t2','msl','apcp']:
                    myvar = f.createVariable(variable,'f4',('time','latitude','longitude'))
                myvar[:] = out[variable]['values']
                myvar.setncattr('long_name', out[variable]['name'])
                myvar.setncattr('units', out[variable]['units'])
            f.Conventions = 'CF-1.8'
            f.version = '1_2023-10-14'
            f.model_name = 'GraphCast'
            f.model_version = 'v1'
            f.initialization_model = 'GFS'
            f.initialization_time = '%s-%s-%sT%s:00:00' % (inityr,initmnth,initday,inithr)
            f.first_forecast_hour = '0'
            f.last_forecast_hour = '240'
            f.forecast_hour_step = '6'
            f.creation_time = (datetime.datetime.utcnow()).strftime('%Y-%m-%dT%H:%M:%S')
            f.close()


    def patch_retrieve_request(self, r):
        if r.get("class", "od") != "od":
            return

        if r.get("type", "an") not in ("an", "fc"):
            return

        if r.get("stream", "oper") not in ("oper", "scda"):
            return

        if self.use_an:
            r["type"] = "an"
        else:
            r["type"] = "fc"

        time = r.get("time", 12)

        r["stream"] = {
            0: "oper",
            6: "scda",
            12: "oper",
            18: "scda",
        }[time]

    def parse_model_args(self, args):
        import argparse

        parser = argparse.ArgumentParser("ai-models graphcast")
        parser.add_argument("--use-an", action="store_true")
        parser.add_argument("--override-constants")
        return parser.parse_args(args)


model = GraphcastModel
