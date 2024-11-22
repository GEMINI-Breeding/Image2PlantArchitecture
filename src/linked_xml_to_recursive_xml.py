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
                est_node_index = -1
                est_petiole_index = -1

                # Recursively call the function for all subelements
                where_am_i_recursive(elem, est_node_index, est_petiole_index, elem)

            elif elem.tag == 'phytomer':
                est_node_index += 1
                est_petiole_index = -1
                where_am_i_recursive(elem[0], est_node_index, est_petiole_index, parent_shoot)
            elif elem.tag == 'petiole':
                #est_petiole_index += 1
                est_petiole_index = 0 # Even if the actual index is calculated, Helios regards the petiol index to 0
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
                    # It will append the shoot after petiole
                    parent_shoot[parent_node_index+5][0].append(deepcopy(subelem)) 

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

def recursive_to_linked(root):
    # Convert the linked xml file to recursive xml file
    root_linked = deepcopy(root)

    # Update parent info
    update_parent_info(root=root_linked)

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
            last_shoot = plant_instance[-1]
            if count_shoot_elements(last_shoot) > 0:
                flatten_shoots_recursive(last_shoot, plant_instance)
            else:
                break

            # Save file for debugging
            if 0:
                # Pretty print the XML
                pretty_xml = pretty_print_xml(root_linked)
                # Save the linked xml file
                with open('src/test_linked.xml', 'w') as f:
                    f.write(pretty_xml)

    return root_linked

if __name__ == "__main__":
    # Read the linked xml file
    tree = ET.parse('data/generated_Nov14_20224/xml/cowpea_0000_day_30.xml')
    root = tree.getroot()
    # print(root.tag)
    root_recursive = linked_to_recursive(root,debug=True)

    # Pretty print the XML
    pretty_xml = pretty_print_xml(root_recursive)

    if where_am_i(root_recursive):
        print("Passed the test")

    # Save the recursive xml file
    with open('src/test_recursive.xml', 'w') as f:
        f.write(pretty_xml)

    # Read the root_linked xml
    randomize_parent(root_recursive)
    pretty_xml = pretty_print_xml(root_recursive)
    # Save the recursive xml file
    with open('src/test_recursive_wrong.xml', 'w') as f:
        f.write(pretty_xml)

    # Save 
    # Pretty print the XML
    pretty_xml = pretty_print_xml(root_recursive)
    # Save the recursive xml file
    with open('src/test_recursive_fixed.xml', 'w') as f:
        f.write(pretty_xml)

    root_linked = recursive_to_linked(root_recursive)

    # Pretty print the XML
    pretty_xml = pretty_print_xml(root_linked)
    # Save the linked xml file
    with open('src/test_linked.xml', 'w') as f:
        f.write(pretty_xml)

