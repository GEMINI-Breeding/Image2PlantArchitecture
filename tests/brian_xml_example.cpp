#include "Visualizer.h"
#include "PlantArchitecture.h"

using namespace helios;

int main()
{

    Context context;

    PlantArchitecture plantarchitecture(&context);

    plantarchitecture.loadPlantModelFromLibrary("cowpea");

    std::ifstream file("../xml/plantstring.txt");
    std::string plantstring;
    std::getline(file, plantstring);
    file.close();

    uint plantID = plantarchitecture.generatePlantFromString(plantstring, plantarchitecture.getCurrentPhytomerParameters());

    plantarchitecture.writePlantStructureXML(plantID, "../xml/cowpea_test.xml");

    Visualizer visualizer(800);

    visualizer.buildContextGeometry(&context);
    visualizer.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);

    visualizer.plotUpdate();

    visualizer.printWindow("../images/cowpea_XMLinput.jpeg");
    visualizer.closeWindow();

    Context context_out;

    PlantArchitecture plantarchitecture_out(&context_out);

    plantarchitecture_out.loadPlantModelFromLibrary("cowpea");

    std::vector<uint> plantIDs = plantarchitecture_out.readPlantStructureXML("../xml/cowpea_test.xml");

    plantarchitecture_out.writePlantStructureXML(plantIDs.front(), "../xml/cowpea_test_out.xml");

    Visualizer visualizer_out(800);

    visualizer_out.buildContextGeometry(&context_out);
    visualizer_out.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);

    visualizer_out.plotUpdate();

    visualizer_out.printWindow("../images/cowpea_XMLoutput.jpeg");
    visualizer_out.closeWindow();
}