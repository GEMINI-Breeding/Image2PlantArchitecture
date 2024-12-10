import xml.etree.ElementTree as ET
import xml.dom.minidom
from copy import deepcopy
import random


def pretty_print_xml(element):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(element, 'UTF-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="\t", encoding="UTF-8").decode('UTF-8')

    # Remove unwanted line changes
    pretty_xml = "\n".join([line for line in pretty_xml.split("\n") if line.strip()])

    return pretty_xml

def randomize_parent(root):
    # Function to recursively print the tag of the elements
    parent_shoot = []
    est_node_index = 0
    est_petiole_index = 0
    test_pass = True
    def where_am_i_recursive(elem, level, parent_shoot):
        print(f"{'  ' * level}Tag: {elem.tag}")
        global est_node_index, est_petiole_index
        global test_pass
        if elem.tag == 'shoot':
            for subelem in elem:
                if subelem.tag == 'parent_shoot_ID':
                    subelem.text = f"{random.randint(1,100)}"
                if subelem.tag == 'parent_node_index':
                    subelem.text = f"{random.randint(1,100)}"
                if subelem.tag == 'parent_petiole_index':
                    subelem.text = f"{random.randint(1,100)}"

        elif elem.tag == 'internode':
            est_node_index += 1
            est_petiole_index = -1
        elif elem.tag == 'petiole':
            est_petiole_index += 1


        # Recursively call the function for all subelements
        for subelem in elem:
            where_am_i_recursive(subelem, level + 1, parent_shoot)
    
    # Start the recursive printing
    where_am_i_recursive(root, 0, parent_shoot)

    return test_pass    

def where_am_i(root):
    # Function to recursively print the tag of the elements
    parent_shoot = []
    est_node_index = 0
    est_petiole_index = 0
    test_pass = True
    def where_am_i_recursive(elem, level, parent_shoot):
        print(f"{'  ' * level}Tag: {elem.tag}")
        global est_node_index, est_petiole_index
        global test_pass
        if elem.tag == 'shoot':
            # Check if it's the first shoot
            if elem.attrib['ID'] == '0':
                pass
            else:
                # Check the parent shoot
                upper_shoot = parent_shoot[-1]
                # Reset indexes
                current_shoot_index = int(elem.attrib['ID'])
                est_parent_shoot_index = int(upper_shoot.attrib['ID'])
                # Check the parent_shoot_ID, parent_node_index, and parent_petiole_index
                for subelem in elem:
                    if subelem.tag == 'parent_shoot_ID':
                        parent_shoot_ID = int(subelem.text.strip())
                    if subelem.tag == 'parent_node_index':
                        parent_node_index = int(subelem.text)
                    if subelem.tag == 'parent_petiole_index':
                        parent_petiole_index = int(subelem.text)
               
                
                if est_parent_shoot_index == parent_shoot_ID:
                    if est_node_index == parent_node_index:
                        if est_petiole_index == parent_petiole_index:
                            print("Nice~")
                    parent_shoot.pop()
                else:
                    print("What?")
                    test_pass = False

            # Append
            parent_shoot.append(elem)

            # Reset indexes
            est_node_index = -1
            est_petiole_index = -1

        elif elem.tag == 'internode':
            est_node_index += 1
            est_petiole_index = -1
        elif elem.tag == 'petiole':
            est_petiole_index += 1


        # Recursively call the function for all subelements
        for subelem in elem:
            where_am_i_recursive(subelem, level + 1, parent_shoot)
    
    # Start the recursive printing
    where_am_i_recursive(root, 0, parent_shoot)

    return test_pass

def update_parent_info(root):
    # Function to recursively print the tag of the elements
    def where_am_i_recursive(root, est_node_index, est_petiole_index, parent_shoot):
        for elem in root:
            if elem.tag == 'shoot':
                # Check if it's the first shoot
                if elem.attrib['ID'] == '0' or parent_shoot == None:
                    # Reset the indexes
                    for subelem in elem:
                        if subelem.tag == 'parent_shoot_ID':
                            subelem.text = f" {-1} "
                        if subelem.tag == 'parent_node_index':
                            subelem.text = f" {0} "
                        if subelem.tag == 'parent_petiole_index':
                            subelem.text = f" {0} "
                    pass
                else:
                    # Check the parent shoot
                    # Reset indexes
                    current_shoot_index = int(elem.attrib['ID'])
                    est_parent_shoot_index = int(parent_shoot.attrib['ID'])
                    # Update
                    for subelem in elem:
                        if subelem.tag == 'parent_shoot_ID':
                            subelem.text = f" {est_parent_shoot_index} "
                        if subelem.tag == 'parent_node_index':
                            subelem.text = f" {est_node_index} "
                        if subelem.tag == 'parent_petiole_index':
                            subelem.text = f" {est_petiole_index} "

                # Reset indexes
                est_node_index = 0
                est_petiole_index = 0

                # Recursively call the function for all subelements
                where_am_i_recursive(elem, est_node_index, est_petiole_index, elem)

            elif elem.tag == 'phytomer':
                # One level differ for internode
                where_am_i_recursive(elem[0], est_node_index, est_petiole_index, parent_shoot)
                est_petiole_index = 0
                est_node_index += 1
            elif elem.tag == 'petiole':
                where_am_i_recursive(elem, est_node_index, est_petiole_index, parent_shoot)
                est_petiole_index += 1
                pass

    
    # Start the recursive printing
    where_am_i_recursive(root[0], -1, -1, None)

    return True


def linked_to_recursive(root,debug=False):
    
    # Convert the linked xml file to recursive xml file
    root_recursive = deepcopy(root)

    # Iterate over plant instances
    for elem in root_recursive:
        plant_ID = int(elem.attrib['ID'])
        while True:
            # Check if the plant has any shoots
            if elem[-1].tag == 'shoot' and elem[-1].attrib['ID'] == '0':
                break
            # Check the last element of the plant
            subelem = elem[-1]
            if subelem.tag == 'shoot':
                # Get the shoot ID attribute
                shoot_ID = int(subelem.attrib['ID'])
                # print("shoot_ID: ", shoot_ID)
                # Check the parent_shoot_ID, parent_node_index, and parent_petiole_index
                for subsubelem in subelem:
                    if subsubelem.tag == 'parent_shoot_ID':
                        parent_shoot_ID = int(subsubelem.text.strip())
                        # print("parent_shoot_ID: ", parent_shoot_ID)
                    if subsubelem.tag == 'parent_node_index':
                        parent_node_index = int(subsubelem.text)
                        # print("parent_node_index: ", parent_node_index)
                    if subsubelem.tag == 'parent_petiole_index':
                        parent_petiole_index = int(subsubelem.text)
                        # print("parent_petiole_index: ", parent_petiole_index)
                
                if parent_shoot_ID == -1:
                    pass
                else:
                    # print(root_recursive[plant_ID][parent_shoot_ID+2][parent_node_index+5][0])
                    # Move the shoot to the corresponding parent shoot, internode, and petiole and remove it from the root
                    # root_recursive[plant_ID][parent_shoot_ID+2][parent_node_index+5][0].append(deepcopy(subelem))
                    # Find the parent shoot based on the parent_shoot_ID, parent_node_index, and parent_petiole_index
                    for shoot in elem.findall('shoot'):
                        if int(shoot.attrib['ID']) == parent_shoot_ID:
                            parent_shoot = shoot
                            break
                    # Move the shoot to the corresponding parent
                    # 5 means index offset from shoot_type_label ~ phytomer
                    # [0] means the internode
                    if 0:
                        # It will append the shoot after petiole
                        parent_shoot[parent_node_index+5][0].append(deepcopy(subelem)) 
                    else:
                        if 1:
                            # Append the shoot on parent_petiole_index
                            parent_shoot[parent_node_index+5][0][parent_petiole_index+4].append(deepcopy(subelem))
                        else:
                            # It will preserve create start - end consistency
                            insert_idx = -1
                            for idx, subsubelem in enumerate(parent_shoot[parent_node_index+5][0][parent_petiole_index+4]):
                                if subsubelem.tag == 'leaf':
                                    insert_idx += 1
                                    break
                            parent_shoot[parent_node_index+5][0][parent_petiole_index+4].insert(insert_idx, deepcopy(subelem)) 

                    # Remove the shoot from the root
                    elem.remove(subelem)

                    # Debug
                    if debug:
                        pretty_xml = pretty_print_xml(root_recursive)
                        # Save the linked xml file
                        with open('src/debug.xml', 'w') as f:
                            f.write(pretty_xml)
            
    return root_recursive

def find_all_nested_elements(root):
    """Find and print all nested elements in the XML tree."""
    for elem in root.iter():
        print(f"Tag: {elem.tag}, Attributes: {elem.attrib}")

def count_shoot_elements(root):
    # Find all elements with the tag 'shoot'
    shoot_elements = root.findall('.//shoot')
    # Return the count of 'shoot' elements
    return len(shoot_elements)

def recursive_to_linked(root, debug=False):
    # Convert the linked xml file to recursive xml file
    root_linked = deepcopy(root)

    # Update parent info
    update_parent_info(root=root_linked)

    # Save file for debugging
    if debug:
        # Pretty print the XML
        pretty_xml = pretty_print_xml(root_linked)
        # Save the linked xml file
        with open('src/debug.xml', 'w') as f:
            f.write(pretty_xml)

    # Function to recursively flatten shoots
    def flatten_shoots_recursive(root, plant_instance_root):
        for elem in root:
            if elem.tag == 'shoot':
                shoot_id = int(elem.attrib['ID'])
                if shoot_id != 0:
                    # Create a new shoot element in the new parent
                    new_shoot = deepcopy(elem)
                    plant_instance_root.append(new_shoot)
                    
                    # Remove the elem
                    root.remove(elem)
                    break
            flatten_shoots_recursive(elem, plant_instance_root)

    # Start the flattening process
    for plant_instance in root_linked:
        # Check if the last shoot of plant instance is nested
        while True:
            all_flat = True
            for elem in plant_instance:
                if elem.tag == 'shoot':
                    shoot_cnt = count_shoot_elements(elem)
                    if shoot_cnt == 0:
                        # This shoot is flattened
                        pass
                    else:
                        all_flat = False
                        flatten_shoots_recursive(elem, plant_instance)
                        break

            if all_flat:
                break

            # Save file for debugging
            if debug:
                # Pretty print the XML
                pretty_xml = pretty_print_xml(root_linked)
                # Save the linked xml file
                with open('src/test_linked.xml', 'w') as f:
                    f.write(pretty_xml)

    return root_linked

if __name__ == "__main__":
    import os
    import numpy as np
    from string_to_xml_to_vec import string2xml, xml2vec, vec2xml, xml2string, string2vec, vec2string
    # Test 1: Read XML and convert to vec, then back to XML and 
    # check if the XML is the same as the original XML
    if 1:
        # Read the XML file
        xml_file = "data/generated_Nov22_2024/xml/cowpea_0097_day_16.xml"
        tree = ET.parse(xml_file)
        root = tree.getroot()

        root = linked_to_recursive(root,debug=True)
        
        # Pretty print the XML
        pretty_xml = pretty_print_xml(root)
        # Save the linked xml file
        with open('src/test_recursive.xml', 'w') as f:
            f.write(pretty_xml)

        plant_array = []
        for plant_instance in root:
            plant_instance_array = []
            xml2vec(plant_instance, plant_instance_array)
            plant_array.append(plant_instance_array)

        # Convert vec to xml
        new_root = ET.Element("helios")
        for i, plant_array in enumerate(plant_array):
            xml_output = vec2xml(plant_array, plant_id=i, debug=True)
            new_root.append(xml_output[0])

        pretty_xml = pretty_print_xml(new_root)
        # Save the linked xml file
        with open('src/debug.xml', 'w') as f:
            f.write(pretty_xml)

        # Update parent
        new_root = recursive_to_linked(root=new_root, debug=True)
        # Pretty print the XML
        pretty_xml = pretty_print_xml(new_root)
        # Save the linked xml file
        with open('src/debug.xml', 'w') as f:
            f.write(pretty_xml)

        

        # Check if the XML is the same as the original XML
        if ET.tostring(new_root) == ET.tostring(root):
            print("The XML is the same as the original XML.")
        else:
            print("The XML is not the same as the original XML.")
            print("Original XML:")
            print(pretty_print_xml(root))
            print("Generated XML:")
            print(pretty_print_xml(new_root))

    # Test 2: Convert string to vec and back to string
    if 0:
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
