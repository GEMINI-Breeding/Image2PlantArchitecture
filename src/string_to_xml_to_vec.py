import xml.etree.ElementTree as ET
import xml.dom.minidom
import re
import os
import numpy as np

from linked_xml_to_recursive_xml import linked_to_recursive, update_parent_info, recursive_to_linked

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
    rough_string = ET.tostring(element, 'UTF-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="\t", encoding="UTF-8").decode('UTF-8')

    # Remove unwanted line changes
    pretty_xml = "\n".join([line for line in pretty_xml.split("\n") if line.strip()])

    return pretty_xml

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


#organ2num = {'shoot': 0, 'internode': 1, 'petiole': 2, 'leaf': 3}
organ2num = {'shoot': 0, 'internode': 1, 'petiole': 2, 
             'leaf0': 3, 'leaf1': 4, 'leaf2':5, 'leaf3':5, 'leaf4':5}
def xml2vec(root, plant_array, depth=0, plant_age=0, leaf_count=0):
    tag = root.tag
    if tag == 'plant_instance':
        # Get attributes
        for subelem in root:
            if subelem.tag == 'base_position':
                # Skip the base position
                continue
            elif subelem.tag == 'plant_age':
                # Save plant_age to global
                plant_age = float(subelem.text)
            elif subelem.tag == 'shoot':
                # Make the first shoot
                xml2vec(subelem, plant_array, depth=depth, plant_age=plant_age)
            
    elif tag == 'shoot':
        # Define the shoot vector
        # shoot = [depth, organ_type, base_pitch, base_yaw, base_roll, plant_age, type], just include plant age to the shoot?
        line = [depth, organ2num['shoot'], 0, 0, 0, plant_age, 0]
        for subsubelem in root:
            if subsubelem.tag == 'shoot_type_label':
                shoot_type = subsubelem.text.strip()
                line[6] = shoottype2num[shoot_type]
            elif subsubelem.tag == 'base_rotation':
                # Parse the base rotation. example:" 8.37393 304.62 189.694 "
                base_rotation = subsubelem.text.strip().split(" ")
                line[2] = float(base_rotation[0])
                line[3] = float(base_rotation[1])
                line[4] = float(base_rotation[2])
                # Append the line to the plant_array
                plant_array.append(line)
            elif subsubelem.tag == 'phytomer':
                # Parse the phytomer
                xml2vec(subsubelem, plant_array, depth=depth,plant_age=plant_age)
            else:
                # Skip the other elements
                pass
    elif tag == 'phytomer':
        for elem in root:
            # Phytomer and internode are at the same depth
            xml2vec(elem, plant_array, depth=depth, plant_age=plant_age)

    elif tag == 'internode':
        # Define the internode vector
        # internode = [depth, organ_type, length, radius, pitch, phyllotactic_angle]
        line = [depth, organ2num['internode'], 0, 0, 0, 0]
        # Get attributes
        for subelem in root:
            if 'length' in subelem.tag:
                line[2] = float(subelem.text)
            elif 'radius' in subelem.tag:
                line[3] = float(subelem.text)
            elif 'pitch' in subelem.tag:
                line[4] = float(subelem.text)
            elif 'phyllotactic_angle' in subelem.tag:
                line[5] = float(subelem.text)
                # Append the line to the plant_array
                plant_array.append(line)
            elif "petiole" in subelem.tag:
                xml2vec(subelem, plant_array, depth=depth,plant_age=plant_age)

    elif tag == 'petiole':
        # Define the petiole vector
        # petiole = [depth, organ_type, length, radius, pitch, petiole_curvature, leaflet_scale]
        line = [depth, organ2num['petiole'], 0, 0, 0, 0, 0]
        leaf_count = 0
        for subelem in root:
            if 'length' in subelem.tag:
                line[2] = float(subelem.text)
            elif 'radius' in subelem.tag:
                line[3] = float(subelem.text)
            elif 'pitch' in subelem.tag:
                line[4] = float(subelem.text)
            elif 'curvature' in subelem.tag:
                line[5] = float(subelem.text)
            elif 'leaflet_scale' in subelem.tag:
                line[6] = float(subelem.text)
                # Append the line to the plant_array
                plant_array.append(line)
            elif "leaf" in subelem.tag:
                if leaf_count < 3:
                    xml2vec(subelem, plant_array, depth=depth,plant_age=plant_age, leaf_count=leaf_count)
                leaf_count += 1
            elif "shoot" in subelem.tag:
                xml2vec(subelem, plant_array, depth=depth+1,plant_age=plant_age)

    elif tag == 'leaf':
        # Define the leaf vector
        # leaf = [depth, organ_type, scale, pitch, yaw, roll]
        line = [depth, organ2num[f'leaf{leaf_count}'], 0, 0, 0, 0]
        for subelem in root:
            if 'scale' in subelem.tag:
                line[2] = float(subelem.text)
            elif 'pitch' in subelem.tag:
                line[3] = float(subelem.text)
            elif 'yaw' in subelem.tag:
                line[4] = float(subelem.text)
            elif 'roll' in subelem.tag:
                line[5] = float(subelem.text)

        # Append the line to the plant_array
        plant_array.append(line)

    return plant_array

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

def add_trait_subelement(elem, trait_name, trait_value):
    trait_elem = ET.SubElement(elem,trait_name)
    trait_elem.text = trait_value

shoot_id = 0
def vec2element(root, plant_array, depth=0, debug=False):
    # print("--------------------")
    # print(pretty_print_xml(root))
    cnt = 0
    last_elem = None
    while len(plant_array) > 0:
        cnt += 1
        if cnt > 2048:
            # Raise an error
            # raise ValueError("Infinite loop")
            # print("Infinite loop, force close the loop")
            break
        
       
        # If depth_line is the same as depth, then add the element to the root
        if len(plant_array) > 0:
            line = plant_array[0]
            depth_line = line[0]
            organ_type = int(line[1])
            organ_name = list(organ2num.keys())[list(organ2num.values()).index(organ_type)]
            organ_name = organ_name.capitalize()
            params = line[2:]
            if organ_name == 'Shoot':
                organ_type = int(line[1])
                organ_name = list(organ2num.keys())[list(organ2num.values()).index(organ_type)]
                current_shoot = ET.SubElement(root, organ_name)
                global shoot_id
                current_shoot.set("ID", f"{shoot_id}")
                shoot_id += 1
                shoot_type = list(shoottype2num.keys())[list(shoottype2num.values()).index(line[6])]
                add_trait_subelement(current_shoot,"shoot_type_label",f" {shoot_type} ")
                add_trait_subelement(current_shoot,"parent_shoot_ID",f" TBD ")
                add_trait_subelement(current_shoot,"parent_node_index",f" TBD ")
                add_trait_subelement(current_shoot,"parent_petiole_index",f" TBD ")
                add_trait_subelement(current_shoot,"base_rotation",f" {line[2]:.6g} {line[3]:.6g} {line[4]:.6g} ")
                plant_array = plant_array[1:]

        if len(plant_array) > 0:
            line = plant_array[0]
            depth_line = line[0]
            organ_type = int(line[1])
            organ_name = list(organ2num.keys())[list(organ2num.values()).index(organ_type)]
            organ_name = organ_name.capitalize()
            params = line[2:]
            if organ_name == 'Internode':
                current_phytomer = ET.SubElement(current_shoot, "phytomer")
                current_internode = ET.SubElement(current_phytomer, "internode")
                add_trait_subelement(current_internode,"internode_length",
                                     f"{max(params[0], 0.0002):.6g}")
                add_trait_subelement(current_internode,"internode_radius",f"{max(params[1],0.0005):.6g}")
                add_trait_subelement(current_internode,"internode_pitch",f"{(params[2]):.6g}")
                add_trait_subelement(current_internode,"internode_phyllotactic_angle",f"{(params[3]):.6g}")
                plant_array = plant_array[1:]

        if len(plant_array) > 0:
            line = plant_array[0]
            depth_line = line[0]
            organ_type = int(line[1])
            organ_name = list(organ2num.keys())[list(organ2num.values()).index(organ_type)]
            organ_name = organ_name.capitalize()
            params = line[2:]
            if organ_name == 'Petiole':
                current_petiole = ET.SubElement(current_internode, "petiole")
                add_trait_subelement(current_petiole,"petiole_length",f"{max(params[0],1e-7):.6g}")
                add_trait_subelement(current_petiole,"petiole_radius",f"{max(params[1],4e-06):.6g}")
                add_trait_subelement(current_petiole,"petiole_pitch",f"{(params[2]):.6g}")
                add_trait_subelement(current_petiole,"petiole_curvature",f"{(params[3]):.6g}")
                add_trait_subelement(current_petiole,"leaflet_scale",f"{(params[4]):.6g}")
                plant_array = plant_array[1:]
                last_elem = 'Petiole'

        if len(plant_array) > 0:
            line = plant_array[0]
            depth_line = line[0]
            organ_type = int(line[1])
            organ_name = list(organ2num.keys())[list(organ2num.values()).index(organ_type)]
            organ_name = organ_name.capitalize()
            params = line[2:]
            if "Leaf" in organ_name:
                current_leaf = ET.SubElement(current_petiole, "leaf")
                add_trait_subelement(current_leaf,"leaf_scale",f"{max(params[0], 0.0002):.6g}")
                add_trait_subelement(current_leaf,"leaf_pitch",f"{(params[1]):.6g}")
                add_trait_subelement(current_leaf,"leaf_yaw",f"{(params[2]):.6g}")
                add_trait_subelement(current_leaf,"leaf_roll",f"{(params[3]):.6g}")
                plant_array = plant_array[1:]

            if debug:
                # Pretty print the XML
                pretty_xml = pretty_print_xml(root)
                # Save the linked xml file
                with open('src/debug.xml', 'w') as f:
                    f.write(pretty_xml)


        if len(plant_array) > 0:
            next_depth = plant_array[0][0]
            if next_depth > depth_line:
                # Subsample the plant_array
                # Find the end index
                # Check if all the lines are the same depth
                depth_list = [arr[0] for arr in plant_array]
                end_idx = len(plant_array)
                if np.all(depth_list == next_depth):
                    plant_array_sub = plant_array
                    plant_array = []
                else:
                    start_idx = 0
                    # Find upper depth
                    for idx in range(start_idx, len(plant_array)):
                        if plant_array[idx][0] < next_depth:
                            end_idx = idx
                            break
                    plant_array_sub = plant_array[start_idx:end_idx]
                    if start_idx > 0:
                        plant_array = plant_array[:start_idx] + plant_array[end_idx:] # ValueError('operands could not be broadcast together with shapes (0,8) (2,8) ')
                    else:
                        plant_array = plant_array[end_idx:]
                    #plant_array = np.concatenate((plant_array, plant_array_sub), axis=0)
                # Process the plant_array_sub
                # if depth_line == 0:

                # else:
                vec2element(current_petiole, plant_array_sub, next_depth,debug=debug)

                if debug:
                    pretty_xml = pretty_print_xml(root)
                    # Save the linked xml file
                    with open('src/debug.xml', 'w') as f:
                        f.write(pretty_xml)
                # print("--------------------")
                # print(pretty_print_xml(root))




def vec2xml(plant_array,  plant_id=0, debug=False):

    root = ET.Element("helios")
    current_plant = ET.SubElement(root, "plant_instance")
    # Set Plant ID
    current_plant.set("ID", str(plant_id))

    line = plant_array[0]
    # Add base_position element
    base_position = ET.SubElement(current_plant, "base_position")
    base_position.text = f" {0} {0} {0} "
    # Add plant_age element
    plant_age = ET.SubElement(current_plant, "plant_age")
    plant_age.text = f" {abs(int(line[5]))} "


    # # Add first shoot
    # organ_type = int(line[1])
    # organ_name = list(organ2num.keys())[list(organ2num.values()).index(organ_type)]
    # current_shoot = ET.SubElement(current_plant, organ_name)
    # current_shoot.set("ID", f"{0}")
    # shoot_type = list(shoottype2num.keys())[list(shoottype2num.values()).index(line[-1])]
    # add_trait_subelement(current_shoot,"shoot_type_label",f" {shoot_type} ")
    # add_trait_subelement(current_shoot,"parent_shoot_ID",f" {-1} ")
    # add_trait_subelement(current_shoot,"parent_node_index",f" {0} ")
    # add_trait_subelement(current_shoot,"parent_petiole_index",f" {0} ")
    # add_trait_subelement(current_shoot,"base_rotation",f" {line[2]} {line[3]} {line[4]} ")
    # plant_array = plant_array[1:]

    if debug:
        # Pretty print the XML
        pretty_xml = pretty_print_xml(root)

        # Save the linked xml file
        with open('src/debug.xml', 'w') as f:
            f.write(pretty_xml)

    

    # Start with depth = 0
    global shoot_id
    shoot_id = 0
    vec2element(current_plant, plant_array, depth=0,debug=debug)

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


def save_plant_string(plant_vec, output_path, idx, suffix=""):
    plant_string = vec2string([plant_vec])
    plant_string_file_name = f"{output_path}/plant_string_{suffix}_{idx}.txt"
    # Create output folder
    os.makedirs(os.path.dirname(plant_string_file_name), exist_ok=True)
    with open(plant_string_file_name, "w") as f:
        f.write(plant_string)
    return plant_string_file_name

if __name__ == "__main__":
    pass