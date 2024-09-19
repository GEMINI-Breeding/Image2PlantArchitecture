import os
import shutil
class plantstring2model:

    def __init__(self, program_path, program_name, background_path=None, display=":10.0"):

        self.program_path = program_path
        self.program_name = program_name

        self.display = display

        self.background_path = background_path

        os.environ["DISPLAY"] = self.display

    def run(self, plantstring_path):
        output_path_name = "output"

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
            
        command += f"{plantstring_path}"
        # Run the command using os.system
        os.system(command)

        # Remove the output folder if exists
        if os.path.exists(output_path_name):
            shutil.rmtree(output_path_name)

        # Move the output dir to src
        shutil.move(os.path.join(self.program_path, output_path_name), "./")


# Test 
if __name__ == "__main__":
    p2m = plantstring2model("src/PlantString2Model/build", "PlantString2Model", display=":11.0")
    p2m.run("plant_string.txt")