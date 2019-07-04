using System;


namespace onnx
{
    class Program
    {
        static void Main(string[] args)
        {
           //onnx.CVGG19.Script(args); 
           //onnx.CYOLOV3.Script(args);
           onnx.CFasterRNN.Script(args);
           Console.WriteLine("Finish");
        }

    }
}
