from opendm import types
from opendm import io
from opendm import system
from PIL import Image
import os
import shutil

class ODMInferenceStage(types.ODM_Stage):
    def process(self, args, outputs):
        tree = outputs['tree']
        reconstruction = outputs['reconstruction']

        # Create inference model output directories
        if not io.dir_exists(tree.infermod):
            system.mkdir_p(tree.infermod)
        
        if not io.dir_exists(tree.infer_image_outputdir):
            system.mkdir_p(tree.infer_image_outputdir)

        input_images = os.path.join(tree.input_images)
        self.InferenceModel(input_images, tree)


    def InferenceModel(self, images, tree):
        for image_name in os.listdir(images):
            print(f"Applying inference to {image_name}")
            current_image_path = os.path.join(images, image_name)
            img = Image.open(current_image_path)
            img = img.convert("RGB")

            d = img.getdata()
            
            new_image = []
            for item in d:
            
                # change all white (also shades of whites)
                # pixels to yellow
                if item[0] in list(range(100, 168)) and item[1] in list(range(102, 148)) and item[2] in list(range(43, 119)):
                    new_image.append((93,50,119))
                else:
                    new_image.append(item)
        
            # update image data
            img.putdata(new_image)
            
            # save new image
            new_image_path = os.path.join(tree.infer_image_outputdir, image_name)
            img.save(new_image_path)