import xml.etree.ElementTree as ET
import xml.dom.minidom
import re
import os
import numpy as np

def parse_shoot_attributes(attributes_str):
    """Parse shoot attributes string and return a dictionary."""
    attributes = re.split(r'[,{}]', attributes_str.strip())
    attributes = [attr for attr in attributes if attr]
    keys = ['base_pitch', 'base_yaw', 'roll_angle', 'gravitropic_curvature', 'type']
    return dict(zip(keys, attributes))


def string2xml(data):
    root = ET.Element("plants")
    current_plant = None
    current_shoot = None
    current_leaf = None
    current_internode = None
    current_petiole = None

    element_stack = []
    prev_phytomer = []
    
    # Process line by line
    for line in data.split("\n"):
        if line.strip() == "":
            continue
        # Parse plant_id
        element_stack = []
        plant_id = line.split("{")[0].strip()
        current_plant = ET.SubElement(root, "plant")
        current_plant.set("id", str(plant_id))
        element_stack.append(current_plant) # Plant and shoot
        # Remove the parsed part
        curly_bracket_start = line.find("{")
        line = line[curly_bracket_start:]
        
        # Parse the rest of the line
        while True:
            # If line start with [ add child to the previous element
            if line.startswith("[") or line.startswith("{"):
                if line.startswith("["):
                    phytomer_shoot = True
                else:
                    phytomer_shoot = False
                curly_bracket_start = line.find("{")
                curly_bracket_end = line.find("}") 
                sub_line = line[curly_bracket_start:curly_bracket_end+1]
                # Check if unifoliate or trifoliate is present
                if "unifoliate" in sub_line or "trifoliate" in sub_line:
                    if phytomer_shoot:
                        # Add to the last phytomer
                        current_shoot = ET.SubElement(current_phytomer, "shoot")
                        prev_phytomer.append(current_phytomer)
                    else:
                        current_shoot = ET.SubElement(element_stack[-1], "shoot")
                    values = parse_shoot_attributes(sub_line)
                    current_shoot.set("base_pitch", values['base_pitch'])
                    current_shoot.set("base_yaw", values['base_yaw'])
                    current_shoot.set("roll_angle", values['roll_angle'])
                    current_shoot.set("gravitropic_curvature", values['gravitropic_curvature'])
                    current_shoot.set("type", values['type'])
                    element_stack.append(current_shoot)
                    # Remove the parsed part
                line = line[curly_bracket_end+1:]
                
            while True:
                if line.startswith("["):
                    break
                bracket_start = line.find("(")
                bracket_end = line.find(")")
                if "Internode" in line[:bracket_end+1]:
                    current_phytomer = ET.SubElement(current_shoot, "phytomer")

                    # Read values from the part
                    values = line.split("Internode(")[1].split(")")[0].split(",")
                    current_internode = ET.SubElement(current_phytomer, "internode")
                    current_internode.set("length", values[0])
                    current_internode.set("radius", values[1])
                    current_internode.set("pitch", values[2])
                    if len(values) > 3:
                        current_internode.set("phyllotactic_angle", values[3])

                    # Remove the parsed part
                    line = line[bracket_end+1:]

                    bracket_start = line.find("(")
                    bracket_end = line.find(")")
                    if "Petiole" in line[:bracket_end+1]:
                        values = line.split("Petiole(")[1].split(")")[0].split(",")
                        current_petiole = ET.SubElement(current_phytomer, "petiole")
                        current_petiole.set("length", values[0])
                        if len(values) == 2:
                            current_petiole.set("pitch", values[1])
                        else:
                            current_petiole.set("radius", values[1])
                            current_petiole.set("pitch", values[2])

                        # Remove the parsed part
                        line = line[bracket_end+1:]

                    bracket_start = line.find("(")
                    bracket_end = line.find(")")
                    if "Leaf" in line[:bracket_end+1]:
                        values = line.split("Leaf(")[1].split(")")[0].split(",")
                        current_leaf = ET.SubElement(current_phytomer, "leaf")
                        current_leaf.set("scale", values[0])
                        current_leaf.set("pitch", values[1])
                        current_leaf.set("yaw", values[2])
                        current_leaf.set("roll", values[3])
                        # Remove the parsed part
                        line = line[bracket_end+1:]
            
                
                # If line start with ] move up to the parent element
                if line.startswith("]"):
                    element_stack.pop()
                    sqare_bracket_start = line.find("]")
                    # Return to the previous element
                    current_shoot = element_stack[-1]
                    current_phytomer = prev_phytomer[-1]
                    prev_phytomer.pop()
                    line = line[sqare_bracket_start+1:]
                    break

            if line == "":
                break
    
    return root

def pretty_print_xml(element):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


shoottype2num = {'unifoliate': 1, 'trifoliate': 3}
def attrib2vec(attrib,max_len = 5):
    # Convert the attrib to a vector
    # attrib = {'length': 0.1, 'width': 0.2, 'area': 0.3, 'perimeter': 0.4, 'curvature': 0.5}
    # vec = [0.1, 0.2, 0.3, 0.4, 0.5]
    vec = []
    for key in attrib:
        if attrib[key] in shoottype2num.keys():
            vec.append(shoottype2num[attrib[key]])
        else:
            vec.append(attrib[key])

    # Pad the vector with zeros
    for i in range(len(vec), max_len):
        vec.append(0)

    # Convert to float
    vec = [float(i) for i in vec]
    return vec


organ2num = {'shoot': 0, 'internode': 1, 'petiole': 2, 'leaf': 3}
def xml2vec(root, plant_array, depth=0):
    for elem in root:
        # Get tag
        tag = elem.tag
        if tag == 'phytomer':
            # phytonmer is a group of organs
            xml2vec(elem, plant_array,depth=depth)
        else:
            organ_type = organ2num[tag]
            line = [depth, organ_type]
            # Get attributes
            attrib = attrib2vec(elem.attrib)

            # Cat to the line
            line += attrib
            # Append the line to the plant_array
            plant_array.append(line)

            xml2vec(elem, plant_array,depth=depth+1)


def xml2string(root, outstring=""):
    # Iterate with the plants
    square_bracket = False
    # Add square bracket if main unifoliate or trifoliate is present
    if "unifoliate" in outstring or "trifoliate" in outstring:
        outstring += "["
        square_bracket = True

    outstring += "{"
    outstring += root.get("base_pitch") + ","
    outstring += root.get("base_yaw") + ","
    outstring += root.get("roll_angle") + ","
    outstring += root.get("gravitropic_curvature") + ","
    outstring += root.get("type") + "}"
    
    # Iterate with the phytomers
    for pyhtomer in root:
        for organ in pyhtomer:
            if organ.tag in ["internode", "petiole", "leaf"]:
                outstring += organ.tag.capitalize() + "("
                for key, value in organ.items():
                    outstring += value + ","
                outstring = outstring[:-1] + ")" # Remove the last "," and close the bracket
            if organ.tag == "shoot":
                outstring = xml2string(organ, outstring)
            
    if square_bracket:
        outstring += "]"
                    

    return outstring



def vec2element(root, plant_array, depth=0):
    # print("--------------------")
    # print(pretty_print_xml(root))
    cnt = 0
    while len(plant_array) > 0:
        cnt += 1
        if cnt > 2048:
            # Raise an error
            # raise ValueError("Infinite loop")
            # print("Infinite loop, force close the loop")
            break
            
        line = plant_array[0]
        depth_line = line[0]
        # If depth_line is the same as depth, then add the element to the root
        if depth_line == depth:
            organ_type = int(line[1])
            organ_name = list(organ2num.keys())[list(organ2num.values()).index(organ_type)]
            organ_name = organ_name.capitalize()
            params = line[2:]
            if organ_name == 'Internode' or organ_name == 'Shoot' :
                if organ_name == 'Internode':
                    current_phytomer = ET.SubElement(root, "phytomer")
                    current_internode = ET.SubElement(current_phytomer, "internode")
                    current_internode.set("length", f"{abs(params[0]):.6f}")
                    current_internode.set("radius", f"{abs(params[1]):.6f}")
                    current_internode.set("pitch", f"{params[2]:.6f}")
                    if params[3] > 0:
                        current_internode.set("phyllotactic_angle", f"{params[3]:.6f}")
                    plant_array = plant_array[1:]

                line = plant_array[0]
                depth_line = line[0]
                organ_type = int(line[1])
                organ_name = list(organ2num.keys())[list(organ2num.values()).index(organ_type)]
                organ_name = organ_name.capitalize()
                params = line[2:]
                
                if organ_name == 'Petiole':
                    current_petiole = ET.SubElement(current_phytomer, "petiole")
                    current_petiole.set("length", f"{abs(params[0]):.6f}")
                    current_petiole.set("radius", f"{params[1]:.6f}")
                    current_petiole.set("pitch", f"{params[2]:.6f}")
                    plant_array = plant_array[1:]

                line = plant_array[0]
                depth_line = line[0]
                organ_type = int(line[1])
                organ_name = list(organ2num.keys())[list(organ2num.values()).index(organ_type)]
                organ_name = organ_name.capitalize()
                params = line[2:]
                if organ_name == 'Leaf':
                    current_leaf = ET.SubElement(current_phytomer, "leaf")
                    current_leaf.set("scale", f"{abs(params[0]):.6f}")
                    current_leaf.set("pitch", f"{params[1]:.6f}")
                    current_leaf.set("yaw", f"{params[2]:.6f}")
                    current_leaf.set("roll", f"{params[3]:.6f}")
                    plant_array = plant_array[1:]

                if len(plant_array) > 0:
                    line = plant_array[0]
                    depth_line = line[0]
                    organ_type = int(line[1])
                    organ_name = list(organ2num.keys())[list(organ2num.values()).index(organ_type)]
                    organ_name = organ_name.capitalize()
                    params = line[2:]
                    if organ_name == 'Shoot':
                        current_shoot = ET.SubElement(current_phytomer, "shoot")
                        current_shoot.set("base_pitch", f"{params[0]:.6f}")
                        current_shoot.set("base_yaw", f"{params[1]:.6f}")
                        current_shoot.set("roll_angle", f"{params[2]:.6f}")
                        current_shoot.set("gravitropic_curvature", f"{params[3]:.6f}")
                        if type(params[4]) != int:
                            # get 1 or 3 based on the value
                            # params[4] = 1 if abs(params[4]-1) < abs(params[4]-3) else 3
                            params[4] = 3 # Hardcode to 3 for other shoots
                        shoot_type = list(shoottype2num.keys())[list(shoottype2num.values()).index(params[4])]
                        current_shoot.set("type", shoot_type)
                        plant_array = plant_array[1:]
            # print("--------------------")
            # print(pretty_print_xml(root))
        elif depth_line < depth:
            pass

        # Check if next line is different from the current depth
        if len(plant_array) > 0:
            next_depth = plant_array[0][0]
            if next_depth > depth_line:
                # Subsample the plant_array
                # Find the end index
                # Check if all the lines are the same depth
                dephts = plant_array[:,0]
                if np.all(dephts == next_depth):
                    plant_array_sub = plant_array
                    plant_array = []
                else:
                    for idx, line in enumerate(plant_array):
                        if line[0] != depth_line:
                            start_idx = idx
                            break
                    # Find the count
                    for idx, line in enumerate(plant_array):
                        if idx < start_idx:
                            continue
                        if line[0] == depth_line:
                            end_idx = idx
                            break
                    plant_array_sub = plant_array[start_idx:end_idx]
                    plant_array = np.concatenate((plant_array[:start_idx], plant_array[end_idx:]), axis=0)
                # Process the plant_array_sub
                vec2element(current_shoot, plant_array_sub, next_depth)


def vec2xml(plant_array,  plant_id=0):

    # Pre-build the structure
    depths = plant_array[:,0]
    max_depth = int(np.max(depths))
    root = ET.Element("plants")
    current_plant = ET.SubElement(root, "plant")
    current_plant.set("id", str(plant_id))

    # Add first shoot
    line = plant_array[0]
    organ_type = int(line[1])
    organ_name = list(organ2num.keys())[list(organ2num.values()).index(organ_type)]
    organ_name = organ_name.capitalize()
    params = line[2:]
    current_shoot = ET.SubElement(current_plant, "shoot")
    current_shoot.set("base_pitch", f"{params[0]:.6f}")
    current_shoot.set("base_yaw", f"{params[1]:.6f}")
    current_shoot.set("roll_angle", f"{params[2]:.6f}")
    current_shoot.set("gravitropic_curvature", f"{params[3]:.6f}")
    # Check the type of params[4]
    if type(params[4]) != int:
        # get 1 or 3 based on the value
        #params[4] = 1 if abs(params[4]-1) < abs(params[4]-3) else 3
        params[4] = 1 # Hardcode to 1 for the first shoot
    shoot_type = list(shoottype2num.keys())[list(shoottype2num.values()).index(params[4])]
    current_shoot.set("type", shoot_type)
    
    plant_array = plant_array[1:]
    # Start with depth = 1
    vec2element(current_shoot, plant_array, depth=1)

    # print(pretty_print_xml(root))

    return root


def string2vec(data_string):
    # Convert string to xml
    xml_output = string2xml(data_string)

    # Convert xml to vec
    total_plant_array = []
    for plant in xml_output:
        plant_array = []
        xml2vec(plant, plant_array)
        total_plant_array.append(plant_array)

    return total_plant_array

def vec2string(plant_array):
    # Convert vec to xml
    new_root = ET.Element("plants")
    for i, plant_array in enumerate(plant_array):
        xml_output = vec2xml(np.array(plant_array), plant_id=i)
        new_root.append(xml_output[0])

    # xml to string
    total_outstring = ""
    for plant in new_root:
        # Reset the outstring
        outstring = ""
        if plant.tag == "plant":
            outstring += plant.get("id") + " "
        outstring += xml2string(plant[0])

        # Append to total_outstring
        total_outstring += outstring
        total_outstring += "\n"

    return total_outstring


def plant_string2words(plant_string):
    """
    Convert a plant string to a list of words.
    
    Args:
    - plant_string (str): The plant string to convert.
    
    Returns:
    - list: The list of words in the plant string.
    """
    # Format the input string with line breaks between components
    formatted_output = plant_string.replace("}Internode", "}\nInternode")\
        .replace(")Internode", ")\nInternode")\
        .replace(")[{", ")\n[{")\
        .replace(")Petiole", ")\nPetiole")\
        .replace(")Leaf", ")\nLeaf")\
        .replace("]", "]\n")
    
    # Split the string into words
    words = formatted_output.split("\n")

    # remove the empty string
    words = [word for word in words if word]

    # Check if the lenth is the same as plant vector
    vec = string2vec(plant_string)[0]
    if len(words) != len(vec):
        print(f"Length of words {len(words)} does not match length of plant vector {len(vec)}")

    return words

if __name__ == "__main__":
    # String to xml
    # Read plant strings from a file

    # Create xml saving directory
    output_dir = 'xml'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = '_data/Syn2Real_cowpea/camA_cowpea_023_004_9562487_plantstring.txt'
    txt_dir = '_data/Syn2Real_cowpea/Syn2Real_cowpea/'
    plantstring_files = [f"{txt_dir}{f}" for f in os.listdir(txt_dir) if f.endswith('_plantstring.txt')]
    plantstring_files.sort()
    for file_name in plantstring_files:
        # Read the string from the file
        with open(file_name, 'r', encoding='utf-8') as f:
            data_string = f.read()

        if 0:
            # Convert string to xml
            output_name = os.path.join(output_dir, file_name.split("/")[-1].split(".")[0] + ".xml")
            # Parse the data and generate XML
            xml_output = string2xml(data_string)
            # print(xml_output)

            # Convert xml to vec
            total_plant_array = []
            for plant in xml_output:
                plant_array = []
                xml2vec(plant, plant_array)
                total_plant_array.append(plant_array)
            
            # Debug
            # for line in total_plant_array[0]:
            #     print(line)
            
            # Now we have the vector representation of the string
            # Check if the vector representation can be converted back to the original string

            # Convert vec to xml
            new_root = ET.Element("plants")
            for i, plant_array in enumerate(total_plant_array):
                xml_output = vec2xml(np.array(plant_array), plant_id=i)
                new_root.append(xml_output[0])            
            
            # xml to string
            total_outstring = ""
            for plant in new_root:
                # Reset the outstring
                outstring = ""
                if plant.tag == "plant":
                    outstring += plant.get("id") + " "
                outstring += xml2string(plant[0])

                # Append to total_outstring
                total_outstring += outstring
                total_outstring += "\n" 
        else:
            # Convert string to vec
            total_plant_array = string2vec(data_string)

            # Convert vec to string
            total_outstring = vec2string(total_plant_array)

        # Check if the string is the same as the original string
        if data_string == total_outstring:
            print(f"{file_name} The string is the same as the original string.")
        else:
            print(f"{file_name} The string is not the same as the original string.")
            print("Original string:")
            print(data_string)
            print("Generated string:")
            print(outstring)

        if 0:
            # Optionally, write the pretty-printed XML to a file
            with open(output_name, 'w', encoding='utf-8') as f:
                f.write(pretty_print_xml(xml_output))
            



