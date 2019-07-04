using System;
using System.Diagnostics;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML;
using System.Drawing;
using System.Linq;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Collections.Generic;

namespace onnx
{

    //   dotnet add package Microsoft.ML --version 1.1.0
    // https://github.com/onnx/models/tree/master/models/image_classification/vgg
    // https://www.kaggle.com/orangutan/keras-vgg19-starter/data
    // https://github.com/dotnet/machinelearning/blob/master/test/Microsoft.ML.Tests/OnnxConversionTest.cs
    // https://github.com/microsoft/onnxruntime/blob/master/docs/CSharp_API.md
    /*
    Preprocessing
The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. The transformation should preferrably happen at preprocessing. Check imagenet_preprocess.py for code. */
    public class CVGG19
    {
        public static void Script(string[] args)
        {

            var images = new List<Tuple<string, string>>
            {
                Tuple.Create( "33c24b1ebb9ee4c6b2714cac928f1803","dingo" ),
                Tuple.Create( "34fd564536bf36b1eb64b7861f13e3b9","dingo" ),
                Tuple.Create( "352893b9735e64f4b0c4faec49f88b0c","dingo" ),
                Tuple.Create( "3535f0697cdc07a7206240c575323065","dingo" ),
                Tuple.Create( "33c24b1ebb9ee4c6b2714cac928f1803","dingo" ),
                Tuple.Create( "34fd564536bf36b1eb64b7861f13e3b9","dingo" ),
                Tuple.Create( "352893b9735e64f4b0c4faec49f88b0c","dingo" ),

                /*Tuple.Create( "0a6c192b96e55e2ca37318919b1ffae6","collie" ),
                Tuple.Create( "0a9832b18d360f50f5b3b2ab4c540ddc","collie" ),
                Tuple.Create( "12de62fb1fa5a48d596428dd5a90184a","collie" ),
                Tuple.Create( "12f4a7e00a5a4fc215f6e0c3fda079c3","collie" ),
                Tuple.Create( "152006ba4c78c907b2fc376a3336ae09","collie" ),
                Tuple.Create( "168e2da635938b82819b8a45bbd3dd0d","collie" ),
                Tuple.Create( "17f5ba5fee5bbebe781009cf8d3e1809","collie" ),
                Tuple.Create( "1a98c368ffb2822653ff93475c73deb7","collie" ),
                Tuple.Create( "1d75b0cf9ad1bd98e84bb5d0d4d3ac0f","collie" )*/

                Tuple.Create( "1f6ae325f91713701c9ae3d8ea6714fd","rottweiler" ),
                Tuple.Create( "23e182419007d6293ae31d6fb4b7076d","rottweiler" ),
                Tuple.Create( "241ab15e33417034deadcbbbaa1a51a4","rottweiler" ),
                Tuple.Create( "245c53103d734d7e49df7a107c97ef54","rottweiler" ),
                Tuple.Create( "2617e4f271598e3e74f7e0f94c862db3","rottweiler" ),
                Tuple.Create( "273c19effa703469145fa4891d374ac3","rottweiler" ),
                Tuple.Create( "286592c4838ad48f89d0e898fefc0172","rottweiler" ),
                Tuple.Create( "2af698f7a9582431f12768ee3878d139","rottweiler" ),
                Tuple.Create( "2d0a562b51b634648b09778ae720a2d4","rottweiler" ),
                Tuple.Create( "40e9358c1fba746471370cc4d920cc8f","rottweiler" ),
            };          
            
           float r_mean = 0.485f, r_std = 0.229f, g_mean = 0.456f, g_std = 0.224f, b_mean = 0.406f, b_std = 0.225f;
           string model_path = @"D:\Adiel\trabajos\ONNX\vgg19-bn.onnx";
           InferenceSession _session = new InferenceSession(model_path);


            foreach(var image in images)
            {
                string image_path = $@"D:\Adiel\trabajos\ONNX\datasets\vgg19\train\{image.Item1}.jpg";
                var bmp = ResizeImage(Bitmap.FromFile(image_path), 224, 224);
                var x = ConvertImageToTensor(bmp, r_mean, r_std, g_mean, g_std, b_mean, b_std) ;
                var input = NamedOnnxValue.CreateFromTensor<float>("data", x);

                var output = _session.Run(new[]{input}).First().AsTensor<float>().ToArray();
                var pred = Array.IndexOf(output, output.Max());
                
                var normalization_factor = output.Select(i => Math.Exp(i)).Sum();
                var probabilities = output.Select(i => Math.Exp(i)/normalization_factor).ToArray();

                Console.WriteLine($"{image.Item1}  {image.Item2} --> {pred}");
            }
           //DisplayMetadata(_session); 
        }

        private static void DisplayMetadata(InferenceSession session)
        {
            Action<System.Collections.Generic.IReadOnlyDictionary<string, NodeMetadata>> func = (System.Collections.Generic.IReadOnlyDictionary<string, NodeMetadata> meta)=>{
                foreach(var m in meta)
                {
                    Console.WriteLine($"Key: {m.Key}");
                    Console.WriteLine($"Dimensions: {m.Value.Dimensions.Length}"); 
                    Console.Write("      ");   
                    foreach(var d in m.Value.Dimensions)
                    {
                        Console.Write($"{d.ToString()}, ");
                    }
                        Console.WriteLine();

                    Console.WriteLine($"Element Type: {m.Value.ElementType.ToString()}");       
                    Console.WriteLine($"Is Tensor: {m.Value.IsTensor.ToString()}");

                    }
            };

           Console.WriteLine($"Input MetaData");
           func(session.InputMetadata);

           Console.WriteLine("");
           Console.WriteLine($"Output MetaData");
           func(session.OutputMetadata);
        }

        public static System.Numerics.Tensors.Tensor<float> ConvertImageToTensor(Bitmap image, float r_mean = 0, float r_std = 1, float g_mean = 0, float g_std = 1, float b_mean = 0, float b_std = 1)
        {
            System.Numerics.Tensors.Tensor<float> data = new System.Numerics.Tensors.DenseTensor<float>(new []{1, 3, image.Width, image.Height});
            for(int x = 0 ; x < image.Width ; x++)
            {
                for(int y = 0 ; y < image.Height; y++)
                {
                    Color c = image.GetPixel(x, y);
                    data[0, 0, x, y] = (c.R/255f - r_mean) / r_std;
                    data[0, 1, x, y] = (c.G/255f - g_mean) / g_std;
                    data[0, 2, x, y] = (c.B/255f - b_mean) / b_std;
                }
            }

            return data;
        }

        private static void RemoveFile(string url)
        {
             if (File.Exists(url)) 
            {
               File.Delete(url);
            }
        }

        private static void RunProc(string url)
        {
            var process = new Process();
            process.StartInfo.UseShellExecute = true;
            process.StartInfo.FileName = url;
            process.Start();
        }

        public static Bitmap ResizeImage(Image image, int width, int height)
{
    var destRect = new Rectangle(0, 0, width, height);
    var destImage = new Bitmap(width, height);

    destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

    using (var graphics = Graphics.FromImage(destImage))
    {
        graphics.CompositingMode = CompositingMode.SourceCopy;
        graphics.CompositingQuality = CompositingQuality.HighQuality;
        graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
        graphics.SmoothingMode = SmoothingMode.HighQuality;
        graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

        using (var wrapMode = new ImageAttributes())
        {
            wrapMode.SetWrapMode(WrapMode.TileFlipXY);
            graphics.DrawImage(image, destRect, 0, 0, image.Width,image.Height, GraphicsUnit.Pixel, wrapMode);
        }
    }

    return destImage;
}

        private static void CreateTextFile(string url, string content)
        {
             if (!File.Exists(url)) 
            {
                // Create a file to write to.
                using (StreamWriter sw = File.CreateText(url)) 
                {
                    sw.WriteLine(content);
                }	
            }
        }
    }
}
