# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import numpy as np
import eccodes
import datetime
from .convert import GRIB_TO_CF, GRIB_TO_XARRAY_PL, GRIB_TO_XARRAY_SFC

LOG = logging.getLogger(__name__)

def save_output_xarray_nc(
    output
):
    output.to_netcdf()
def save_output_xarray(
    *,
    output,
    target_variables,
    write,
    all_fields,
    ordering,
    lead_time,
    hour_steps,
    lagged,
    inputdata,
    initdate,
    inittime
):
    LOG.info("Converting output xarray to GRIB and saving")
    output["total_precipitation_6hr"] = output.data_vars[
        "total_precipitation_6hr"
    ].cumsum(dim="time")
    all_fields = all_fields.order_by(
        valid_datetime="descending",
        param_level=ordering,
        remapping={"param_level": "{param}{levelist}"},
    )
    year = str(initdate)[0:4]
    month = str(initdate)[4:6]
    day = str(initdate)[6:8]
    hh = str(inittime)[0:2]
    initdt = datetime.datetime.strptime(f"{year}{month}{day}{hh}","%Y%m%d%H")
    inityr = str(initdt.year)
    initmnth = str(initdt.month).zfill(2)
    initday = str(initdt.day).zfill(2)
    inithr = str(initdt.hour).zfill(2)

    for fs in all_fields:
        param, level = fs["shortName"], fs["level"]
        if level != 0:
            param = GRIB_TO_XARRAY_PL.get(param, param)
            if param not in target_variables:
                continue
            values = inputdata.sel(level=level).data_vars[param].values[0,1]
            
        else:
            param = GRIB_TO_CF.get(param, param)
            param = GRIB_TO_XARRAY_SFC.get(param, param)
            if param not in target_variables or param=='total_precipitation_6hr':
                continue
            values = inputdata.data_vars[param].values[0,1]
        # We want to field north=>south

        values = np.flipud(values.reshape(fs.shape))
        fscopy = fs
        eccodes.codes_set(fscopy.handle.handle, "date", int("%s%s%s" % (year,month,day)))
        eccodes.codes_set(fscopy.handle.handle, "time", int(f"{hh}00"))
        write(
            values,
            template=fscopy,
            step=0,
        )

    for time in range(lead_time // hour_steps):
        for fs in all_fields:
            param, level = fs["shortName"], fs["level"]
            if level != 0:
                param = GRIB_TO_XARRAY_PL.get(param, param)
                if param not in target_variables:
                    continue
                values = output.isel(time=time).sel(level=level).data_vars[param].values
            else:
                param = GRIB_TO_CF.get(param, param)
                param = GRIB_TO_XARRAY_SFC.get(param, param)
                if param not in target_variables:
                    continue
                values = output.isel(time=time).data_vars[param].values

            # We want to field north=>south

            values = np.flipud(values.reshape(fs.shape))
            fscopy = fs
            eccodes.codes_set(fscopy.handle.handle, "date", int("%s%s%s" % (year,month,day)))
            eccodes.codes_set(fscopy.handle.handle, "time", int(f"{hh}00"))
            if param == "total_precipitation_6hr":
                write(
                    values,
                    template=fscopy,
                    startStep=0,
                    endStep=(time + 1) * hour_steps,
                )
            else:
                write(
                    values,
                    template=fscopy,
                    step=(time + 1) * hour_steps,
                )

