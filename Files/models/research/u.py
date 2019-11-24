import os
directory = r"C:\Users\Manta\Downloads\models-master\models-master\research\object_detection\protos"
for file in os.listdir(directory):
    if file.endswith(".proto"):
        os.system("protoc "+directory+"/"+file+" --python_out=. --proto_path="+directory)
