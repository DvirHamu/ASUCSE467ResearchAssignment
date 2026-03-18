import os

from PIL import Image



src = "Dataset/Faces"

dst = "train_faces"



os.makedirs(dst, exist_ok=True)



count = 0



for person in os.listdir(src):

    person_dir = os.path.join(src, person)



    if not os.path.isdir(person_dir):

        continue



    for img in os.listdir(person_dir):



        path = os.path.join(person_dir, img)



        try:

            im = Image.open(path).convert("RGB")

            im = im.resize((512,512))



            save_path = os.path.join(dst, f"{count}.jpg")

            im.save(save_path)



            count += 1



        except:

            continue



print("Total images:", count)
