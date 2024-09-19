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
    std::string plant_string_file = "plantstring.txt";
    bool debug = false;
    bool grow = false;
    bool rotation_view = false;
    float height = 0;
    std::string tile_file = "plugins/visualizer/textures/dirt.jpg";

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg =="-r") {
            rotation_view = true;
        } else if (arg == "-g") {
            grow = true;
        } else if (arg == "-d") {
            debug = true;
        } else if (arg == "-h" && i + 1 < argc) {
            height = std::stof(argv[i+1]);
        } else if (arg == "-tile" && i + 1 < argc) {
            tile_file = argv[i+1];
        } else {
            plant_string_file = arg;
        }
    }

    std::ifstream file(plant_string_file);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << plant_string_file << std::endl;
        return 1;
    }

    // Output the parsed flags for debugging purposes
    std::cout << "Debug: " << (debug ? "true" : "false") << std::endl;
    std::cout << "Grow: " << (grow ? "true" : "false") << std::endl;
    std::cout << "View height" << height << "m" << std::endl;
    if (!tile_file.empty()) {
        std::cout << "Tile file: " << tile_file << std::endl;
    }
    std::cout << "Plant string file: " << plant_string_file << std::endl;


    // Create a save directory if it does not exist
    std::string save_dir = "output";
    std::string command = "mkdir -p " + save_dir;

    if (system(command.c_str()) == -1) {
        std::cerr << "Error creating save directory: " << save_dir << std::endl;
        return 1;
    }



    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string plantstring = buffer.str();
    
    // Remove plant ID if start with <int> and end with " "
    if (plantstring[0] >= '0' && plantstring[0] <= '9') {
        size_t pos = plantstring.find(" ");
        plantstring.erase(0, pos + 1);
    }
    // Print input plant string
    Context context;

    context.seedRandomGenerator(60);
    
    // Add a ground surface with a center position of (0,0,0) and size of row_spacing x plant_spacing
    // Check if tile_file is not none
    if(tile_file != "none"){
        std::vector<uint> UUIDs_ground = context.addTile(make_vec3(0, 0, 0), make_vec2(2, 2), nullrotation, make_int2(2,2),tile_file.c_str());
    }
    

    PlantArchitecture plantarchitecture(&context);

    plantarchitecture.loadPlantModelFromLibrary("cowpea");
    plantarchitecture.buildPlantInstanceFromLibrary(nullorigin, 0);
    std::map<std::string,ShootParameters> shoot_parameters = plantarchitecture.getCurrentShootParameters();
    std::map<std::string, PhytomerParameters> phytomer_parameters;
    phytomer_parameters["unifoliate"] = shoot_parameters.at("unifoliate").phytomer_parameters;
    phytomer_parameters["trifoliate"] = shoot_parameters.at("trifoliate").phytomer_parameters;

    //plantarchitecture.generatePlantFromString(plantstring, phytomer_parameters, nullorigin); 
    plantarchitecture.generatePlantFromString(plantstring, phytomer_parameters); 

    Visualizer vis(1200);
    vis.buildContextGeometry(&context);
    vis.hideWatermark();
    vis.disableMessages();
    vis.setLightingModel(Visualizer::LIGHTING_PHONG);

    float x = 0;
    float y = 0;
    float z = 1.0;

    if(height > 0){
        z = height;
    }
    vis.setCameraPosition(make_vec3(x,y,z), make_vec3(0, 0, 0));
    // Bug: Have to update twice to get the image
    vis.plotUpdate(true);
    vis.plotUpdate(true);
    //vis.plotDepthMap();

    std::stringstream framefile;
    // Generate output file name by replacing .txt with .jpeg
    // Get the file name only
    std::string name_only = plant_string_file.substr(plant_string_file.find_last_of("/\\") + 1);
    name_only = name_only.substr(0, name_only.size() - 4); // Remove .txt and add .jpeg
    std::string save_name = name_only + "_top.jpeg"; // Remove .txt and add "_top.jpeg");    
    // Save to save dir
    std::string save_path = save_dir + "/" + save_name;
    vis.printWindow(save_path.c_str());
    

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
            framefile << name_only << "_" << i << ".jpeg"; // Append angle index to filename

            // Save to save dir
            std::string save_path = save_dir + "/" + framefile.str();
            vis.printWindow(save_path.c_str());
        }
    }

    if (grow) {
        // PlantArchitecture plantarchitecture_grow(&context);
        // plantarchitecture_grow.loadPlantModelFromLibrary("cowpea");
        // plantarchitecture_grow.buildPlantInstanceFromLibrary(nullorigin, 0);
        // Grow the plant for 10 days
        for (int i = 0; i < 20; ++i) {
            plantarchitecture.advanceTime(2);
            vis.buildContextGeometry(&context);
            vis.plotUpdate(true);
            vis.plotUpdate(true); // Update twice due to the mentioned bug
            
            // Generate output file name by replacing .txt with _angle.jpeg to differentiate between images
            std::stringstream framefile;
            framefile << name_only << "_day" << i << ".jpeg"; // Append angle index to filename

            // Save to save dir
            std::string save_path = save_dir + "/" + framefile.str();
            vis.printWindow(save_path.c_str());
        }
    }

    if (debug) {
        vis.plotInteractive();
    }

    return 0;
}