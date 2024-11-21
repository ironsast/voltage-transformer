#4pbysztzrTImPEfHypPu
import roboflow

rf = roboflow.Roboflow(api_key="4pbysztzrTImPEfHypPu")
project = rf.workspace("ironsast-oyacw").project("density-relay")

#can specify weights_filename, default is "weights/best.pt"
version = project.version("2")

#example1 - directory path is "training1/model1.pt" for yolov8 model
version.deploy("yolov11", "C:\GITS\density relay.v1i.yolov11\runs\detect\train\weights", "best.pt")
