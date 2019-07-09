# MachineLearningRepository
<b>Installation</b><br/>
> git clone https://github.com/adiel2012/MachineLearningRepository.git<br/>
> cd MachineLearningRepository<br/>
> pip install virtualenv<br/>
> python -m venv virtual_enviroment<br/>
> .\virtual_enviroment\Scripts\activate.bat<br/>
> python -m pip install --upgrade pip<br/>
> pip install tensorflow<br/>
> pip install --upgrade setuptools<br/>
> pip install tf-nightly<br/>
> pip install keras<br/>
> pip install pillow<br/>
> pip install onnxruntime<br/>
> pip install keras2onnx

<b>Running cifar10 simple example</b><br/>
> cd py<br/>
> experimenting\cifar10example.py<br/>
It creates and train a model of CNN and save it at '..//onnx_models//cifar10X.onnx'<br/>

<b>Running cifar10 trainning using images from folder</b><br/>
> cd py<br/>
> python .\experimenting\cifar10FromFolder.py<br/>
It creates and train a model of CNN and save it at '..//onnx_models//cifar10model1.onnx'<br/>

<b>Running cifar10 trainning with inception v3</b><br/>
> cd py<br/>
> python .\experimenting\train_inception_v3.py<br/>
It creates and train a model of CNN and save it at '..//onnx_models//cifar10_inception_v3.onnx'<br/>

<b>.NetCore 3.0 app using cifar10X.onnx</b><br/>
To be included<br/>

<b>Object detection with Yolov3(trainning and using model)</b><br/>
To be included<br/>

<b>Natural language processing (NLP)</b><br/>
> cd cd java/StanfordNPLCore/corenpl1<br/>
> mvn clean compile assembly:single<br/>
> java -jar acmartifactid-1.0-SNAPSHOT-jar-with-dependencies.jar "paragraph one" "paragraph 2"<br/>