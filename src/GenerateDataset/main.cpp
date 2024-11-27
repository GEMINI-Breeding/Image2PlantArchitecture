#include "PlantArchitecture.h"
#include "Visualizer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib> // For setenv
using namespace helios;


void printUsage(const char* programName) {
    std::cerr << "Usage: " << programName << " -[r] [-g] [-d] [-h <height_m>][-tile <file>] <plant_string_file>" << std::endl;
}

int main(int argc, char* argv[]){
    std::string save_dir = "output";
    bool debug = false;
    bool save_xml = false;
    bool grow = false;
    bool rotation_view = false;
    float height = 0;
    std::string tile_file = "plugins/visualizer/textures/dirt.jpg";
    std::string plant_model_file = "../plant_0.xml";
    std::string output_name = "cowpea";
    uint seed = 60; // Default seed value
    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-r") {
            rotation_view = true;
        } else if (arg == "-g") {
            grow = true;
        } else if (arg == "-d") {
            debug = true;
        } else if (arg == "-xml") {
            save_xml = true;
        } else if (arg == "-h" && i + 1 < argc) {
            height = std::stof(argv[++i]); // i를 증가시켜 다음 값을 건너뜁니다
        } else if (arg == "-tile" && i + 1 < argc) {
            tile_file = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            save_dir = argv[++i];
            printf("Save dir: %s\n", save_dir.c_str());
        } else if (arg == "-f" && i + 1 < argc) {
            plant_model_file = argv[++i];
        } else if (arg == "-seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
            printf("Seed: %u\n", seed);
        } else if (arg == "-name" && i + 1 < argc) {
            output_name = argv[++i];
            printf("Output name: %s\n", output_name.c_str());
        } else {
            printf("Unknown argument: %s\n", arg.c_str());
        }
    }

    // Output the parsed flags for debugging purposes
    std::cout << "Debug: " << (debug ? "true" : "false") << std::endl;
    std::cout << "Grow: " << (grow ? "true" : "false") << std::endl;
    std::cout << "View height: " << height << "m" << std::endl;
    if (!tile_file.empty()) {
        std::cout << "Tile file: " << tile_file << std::endl;
    }

    // Create a save directory if it does not exist
    std::string command = "mkdir -p " + save_dir;

    if (system(command.c_str()) == -1) {
        std::cerr << "Error creating save directory: " << save_dir << std::endl;
        return 1;
    }

    // Print input plant string
    Context context;
    context.seedRandomGenerator(seed);
    // Add a ground surface with a center position of (0,0,0) and size of row_spacing x plant_spacing
    // Check if tile_file is not none
    if (tile_file=="black"){
        std::vector<uint> UUIDs_ground = context.addTile(make_vec3(0, 0, 0), make_vec2(3, 3), nullrotation, make_int2(3,3), RGB::black);
    }else if(tile_file != "none"){
        std::vector<uint> UUIDs_ground = context.addTile(make_vec3(0, 0, 0), make_vec2(3, 3), nullrotation, make_int2(3,3),tile_file.c_str());
    }
    
    PlantArchitecture plantarchitecture(&context);
    plantarchitecture.loadPlantModelFromLibrary("cowpea");
    auto nullorigin = make_vec3(0, 0, 0);
    uint plantID;

    // Check if plant_model_file is not empty
    if(plant_model_file.empty()){
        plantID = plantarchitecture.buildPlantInstanceFromLibrary(nullorigin, 0);
    }else{
        std::vector<uint> plantIDs = plantarchitecture.readPlantStructureXML(plant_model_file);
        plantID = plantIDs.front();
    }


    Visualizer vis(1200);
    vis.clearGeometry();
    vis.buildContextGeometry(&context);
    vis.hideWatermark();
    vis.disableMessages();
    vis.setLightingModel(Visualizer::LIGHTING_PHONG);

    // Set the camera position
    float x = 0;
    float y = 0;
    float z = 1.0;

    if(height > 0){
        z = height;
    }
    vis.setCameraPosition(make_vec3(x,y,z), make_vec3(0, 0, 0));
    vis.plotUpdate(true);
    vis.plotUpdate(true);
    
    // // Save xml
    // if (save_xml)
    // {
    //     // Write the plant structure to an XML file
    //     std::string xml_file = output_name + ".xml";
    //     plantarchitecture.writePlantStructureXML(plantID, xml_file);
    // }
    if(~plant_model_file.empty()){
        std::string output_file = save_dir + "/" + output_name + ".jpeg";
        vis.printWindow(output_file.c_str());
    }
    
    if (rotation_view)
    {
        // Assuming you want to rotate the camera around the origin (0,0,0) in a circular path
        // and save images for each position. Let's do this for a full 360 degrees rotation.
        const float min_radius = 0.3;               // Minimum distance from the origin (closest zoom) 0.5
        const float max_radius = 1.0;               // Maximum distance from the origin (farthest zoom) 1.2
        const float view_angle = 30;                // Field of view angle in degrees, 60
        const int num_steps = 72;                   // Number of steps in the rotation, adjust for more/less images
        const float step_angle = 360.0 / num_steps; // Angle step in degrees

        for (int i = 0; i < num_steps; ++i)
        {
            float angle = step_angle * i * (M_PI / 180.0); // Convert angle to radians
            // Dynamically adjust the radius to zoom in and out
            // Using a sine function to smoothly transition the radius for a cyclic zoom effect
            float radius = min_radius + (sin(angle * 2) + 1) / 2 * (max_radius - min_radius);

            // Calculate x, y, z positions on a circle around the origin at the current radius
            float x = radius * cos(angle);
            float y = radius * sin(angle);
            float z = radius * 1.2; // Adjust z based on the radius to maintain perspective

            vis.setCameraPosition(make_vec3(x, y, z), make_vec3(0, 0, 0));
            vis.plotUpdate(true);
            vis.plotUpdate(true); // Update twice due to the mentioned bug

            // Generate output file name by replacing .txt with _angle.jpeg to differentiate between images
            std::stringstream framefile;
            framefile << output_name << "_" << i << ".jpeg"; // Append angle index to filename

            // Save to save dir
            std::string save_path = save_dir + "/" + framefile.str();
            vis.printWindow(save_path.c_str());
        }
    }

    if (grow) {
        // Grow the plant for 20 days
        float accum_day = 0;
        for (int i = 0; i < 20; ++i) {
            float dt = 1.0;
            vis.clearGeometry();
            if(i > 0){
                plantarchitecture.advanceTime(plantID, dt);
            }
            //accum_day += dt;
            accum_day = plantarchitecture.getPlantAge(plantID);
            vis.buildContextGeometry(&context);
            //vis.plotUpdate(true);
            vis.plotUpdate(true); // Update twice due to the mentioned bug
            
            // Generate output file name by replacing .txt with _angle.jpeg to differentiate between images
            std::stringstream framefile;
            // Convert day to secs
            int secs = accum_day * 24 * 60 * 60;
            //framefile << name_only << "_time_" << std::setfill('0') << std::setw(8) << secs << ".jpeg"; // Append angle index to filename
            framefile << output_name << "_day_" 
                    << std::setfill('0') << std::setw(2)
                    << accum_day << ".jpeg"; // Append angle index to filename

            // Save to save dir
            std::string save_path = save_dir + "/" + framefile.str();
            vis.printWindow(save_path.c_str());

            // Save xml
            if (save_xml)
            {
                // Write the plant structure to an XML file
                std::string xml_file = save_path.replace(save_path.find(".jpeg"), 5, ".xml");
                plantarchitecture.writePlantStructureXML(plantID, xml_file);
            }
        }
    }

    if (debug) {
        vis.plotInteractive();
    }

    return 0;
}