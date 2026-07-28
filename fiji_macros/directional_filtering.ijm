// Select the preprocessed folder
inputDir = "/PATH-TO-YOUR-REPO/blackgold-myelin-pipeline/outputs/1_cortical_fibers/preprocessed/";

list = getFileList(inputDir);

for (i = 0; i < list.length; i++) {
    filename = list[i];

    if (endsWith(filename, "_cortex_inverted.tif")) {

        open(inputDir + filename);

        // Run MorphoLibJ Directional Filtering
        run("Directional Filtering", 
            "type=Max operation=Opening line=50 direction=5");

        // Build output filename
        outputName = replace(filename, "_cortex_inverted.tif", "_directional_filtered.tif");

        // Save result
        saveAs("Tiff", inputDir + outputName);

        close();
    }
}

print("Directional filtering completed.");