# Available modules

A [`SomersModel`](../api_reference/somers_model.rst) consists of different `Modules` which are initialized and run in a specific order. These contain a `GroundwaterModule`, `SoilTemperatureModule` and a `SoilMoistureModule`, each belonging to PeatParcel2D and an `AapModule`, belonging to AAP. Below, an overview is given of the available modules.

## Groundwater modules

| Module | Description |
| ------ | ----------- |
| [Modflow](../api_reference/modflow_module.rst) | Simulate a phreatic head in PeatParcel2D with Modflow 6 in a 2D groundwater model. |
| [Measurements](../api_reference/measurements.rst) | Use a "measured" phreatic head in PeatParcel2D. |


## Soil temperature modules
| Module | Description |
| ------ | ----------- |
| [Fft](../api_reference/fft.rst) | Simulate soil temperature in PeatParcel2D with a Fast Fourier Transform method. |


## Soil moisture modules
| Module | Description |
| ------ | ----------- |
| [DynamicMoisture](../api_reference/dynamic_moisture.rst) | Simulate soil moisture in PeatParcel2D using a lookup table like method for different soil types. |


## Greenhouse gas emission modules
| Module | Description |
| ------ | ----------- |
| [Aap](../api_reference/aap_module.rst) | Simulate Anaerobic Decomposition Potential to calculate CO2 emissions. |
| [Methane](../api_reference/methane_module.rst) | Simulate emission of methane based on ground water levels. |
