import os
import shutil
import subprocess
class plantstring2model:

    def __init__(self, program_path, program_name, background_path=None, display=":10.0", height=2.0, verbose=False):
        

        self.program_path = program_path
        self.program_name = program_name

        self.display = display

        self.background_path = background_path
        self.height = height
        os.environ["DISPLAY"] = self.display

        self.verbose = verbose

        self.output_path_name = "output"

        # Create the output directory
        os.makedirs(self.output_path_name, exist_ok=True)

    def run(self, in_plantstring_path, output_path=None):
        
        plantstring_name = os.path.splitext(os.path.basename(in_plantstring_path))[0]
        if output_path:
            output_file_name = output_path
        else:
            output_file_name = f"{plantstring_name}_top.jpeg"
        # Construct the command
        command = ""
        command += f"cd {self.program_path} && ./{self.program_name} "
        # If self.background_path exists
        if self.background_path:
            if self.background_path == "none":
                # Do not copy 
                pass
            else:
                shutil.copy(self.background_path, self.program_path)

            file_name = os.path.basename(self.background_path)
            # Copy the background tile to build
            command += f"-tile {file_name} "
            
        # Add height
        command += f"-h {self.height} "
        
        # Add the plantstring path
        command += f"-f {in_plantstring_path} "
        command += f"-o {output_file_name} "

        if self.verbose == False:
            command += " > log.txt 2>&1"
            
        # Run the command using os.system
        # os.system(f"{command}")
        # Replace os.system(f"{command}") with subprocess.run
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Check if the command was successful
        if result.returncode == 0:
            # print("Command executed successfully")
            # print(result.stdout)  # Print the standard output
            pass
        else:
            # print("Command failed")
            # print(result.stderr)  # Print the error output
            pass
            
# Test 
if __name__ == "__main__":
    p2m = plantstring2model("src/PlantString2Model/build", "PlantString2Model", display=":11.0")
    p2m.run("/home/lion397/codes/Image2PlantArchitecture/src/plant_string.txt")