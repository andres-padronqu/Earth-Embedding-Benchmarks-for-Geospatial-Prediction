// ==========================================
// PRITHVI - SENTINEL-2 PATCH EXTRACTION
// ==========================================
//
// Description:
// This script exports Sentinel-2 GeoTIFF patches
// for the Prithvi Foundation Model.
//
// Output:
// One 224 x 224 Sentinel-2 patch per city.
//
// Bands:
// B2, B3, B4, B8, B11, B12
//
// ==========================================


// ------------------------------------------
// 1. Cities
// ------------------------------------------

var cities = ee.FeatureCollection([

  ee.Feature(
    ee.Geometry.Point([-99.1332, 19.4326]),
    {city: "cdmx"}
  ),

  ee.Feature(
    ee.Geometry.Point([-103.3496, 20.6597]),
    {city: "guadalajara"}
  ),

  ee.Feature(
    ee.Geometry.Point([-100.3161, 25.6866]),
    {city: "monterrey"}
  ),

  ee.Feature(
    ee.Geometry.Point([-89.5926, 20.9674]),
    {city: "merida"}
  ),

  ee.Feature(
    ee.Geometry.Point([-96.7216, 17.0732]),
    {city: "oaxaca"}
  )

]);


// ------------------------------------------
// 2. Sentinel-2 Collection
// ------------------------------------------

var image = ee.ImageCollection(
    "COPERNICUS/S2_SR_HARMONIZED"
)

.filterDate(
    "2025-01-01",
    "2025-12-31"
)

.filter(
    ee.Filter.lt(
        "CLOUDY_PIXEL_PERCENTAGE",
        20
    )
)

.median()

.select([
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12"
]);


// ------------------------------------------
// 3. Display
// ------------------------------------------

Map.centerObject(cities, 5);

Map.addLayer(
    cities,
    {},
    "Cities"
);


// ------------------------------------------
// 4. Export patches
// ------------------------------------------

cities.evaluate(function(fc){

    fc.features.forEach(function(feature){

        var city = feature.properties.city;

        var point = ee.Geometry.Point(
            feature.geometry.coordinates
        );

        var region = point
            .buffer(1120)
            .bounds();

        Export.image.toDrive({

            image: image.clip(region),

            description:
                city + "_prithvi_patch",

            folder:
                "prithvi",

            fileNamePrefix:
                city + "_prithvi_patch",

            region: region,

            scale: 10,

            crs: "EPSG:4326",

            fileFormat: "GeoTIFF",

            maxPixels: 1e13

        });

    });

});