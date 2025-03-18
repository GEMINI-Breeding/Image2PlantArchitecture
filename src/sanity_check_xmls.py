    if 0:
        # Sanity check XML
        import xml.etree.ElementTree as ET
        from tqdm import tqdm
        xml_path = os.path.join(dataset_dir, "xml")
        xml_files = [x for x in os.listdir(xml_path) if x.endswith("xml")]
        xml_files = [os.path.join(xml_path, xml_file) for xml_file in xml_files]
        xml_files.sort()
        # Add some problematic files
        xml_files.insert(0, "data/20250311_Sideview_40Days/xml/cowpea_6662_day_22.xml")
        print("Sanity check XML files...")
        for xml_file in tqdm(xml_files):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
            except Exception as e:
                print(xml_file)
                print(e)
                raise e