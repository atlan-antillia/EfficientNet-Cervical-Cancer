# Copyright 2022 (C) antillia.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# 2022/10/13 Copyright (C) antillia.com

#Malignant_Lymphoma

import os, sys
import shutil

from PIL import Image
import glob
import traceback

def resize(base_image_dir, output_image_dir, image_size=512):

  classes = ["High squamous intra-epithelial lesion",
            "Low squamous intra-epithelial lesion",
            "Negative for Intraepithelial malignancy",
            "Squamous cell carcinoma"]
  classes = ["HSIL", "LSIL", "NL", "SCC"]

  for cls in classes:
    class_dir = os.path.join(base_image_dir, cls)
    files = glob.glob(class_dir  + "/*.jpg")
    output_dir = os.path.join(output_image_dir, cls)
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
      print("--- Removed {}".format(output_dir))
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    for file in files:
        print("file : {} ".format( file))

        basename = os.path.basename(file)
        name     = basename.split(".")[0]
        
        outfile = name +  ".jpg"
        image = Image.open(file)
        image = image.convert("RGB")
        W, H,  = image.size
        print("W:{}, H:{}".format(W, H))
        WIDTH = H        
        # crop((left, upper, right, lower))
        left   = int( (W - WIDTH)/2.0 )
        upper  = 0
        right  = left + WIDTH
        lower = upper + WIDTH
        print(" left {} upper {} right {} lower {}".format(left, upper, right, lower))
        cropped_image = image.crop((left, upper, right, lower))

        resized_image = cropped_image.resize((image_size, image_size))
        
        output_file = os.path.join(output_dir, outfile)
        resized_image.save(output_file, quality=90)
        print("--- Saved {}".format(output_file))
        


if __name__ == "__main__":
  base_image_dir   =  "./Liquid-based-cytology-pap-smear-images"
  output_image_dir = "./Image_dataset_512x512_master"
  try:

    resize(base_image_dir, output_image_dir, image_size=512)

  except:
    traceback.print_exc()

