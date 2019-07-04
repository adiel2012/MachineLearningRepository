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

//   https://github.com/onnx/models/tree/master/yolov3

namespace onnx
{
    public class CFasterRNN
    {
        public static void Script(string[] args)
        {
           string model_path = @"D:\Adiel\trabajos\ONNX\faster_rcnn_R_50_FPN_1x.onnx";
           InferenceSession _session = new InferenceSession(model_path);

            int w = 800, h = 800;
            string image_path = $@"./images/yolov3/perros2.jpg";
            var bmp = ResizeImage(Bitmap.FromFile(image_path), 32*(w/32+1), 32*(h/32+1));
            float r_mean = 102.9801f, r_std = 1f, g_mean = 115.9465f, g_std = 1f, b_mean = 122.7717f, b_std = 1f;
            var x = ConvertImageToTensor(bmp, r_mean, r_std, g_mean, g_std, b_mean, b_std) ;
            var input = NamedOnnxValue.CreateFromTensor<float>("image", x);

            var output = _session.Run(new[]{input});
                      
        }


        public static System.Numerics.Tensors.Tensor<float> ConvertImageToTensor(Bitmap image, float r_mean = 0, float r_std = 1, float g_mean = 0, float g_std = 1, float b_mean = 0, float b_std = 1)
        {
            System.Numerics.Tensors.Tensor<float> data = new System.Numerics.Tensors.DenseTensor<float>(new []{ 3, image.Height, image.Width});
            for(int x = 0 ; x < image.Width ; x++)
            {
                for(int y = 0 ; y < image.Height; y++)
                {
                    Color c = image.GetPixel(x, y);
                    data[0, 0, x, y] = (c.R - r_mean) / r_std;
                    data[0, 1, x, y] = (c.G - g_mean) / g_std;
                    data[0, 2, x, y] = (c.B - b_mean) / b_std;
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
