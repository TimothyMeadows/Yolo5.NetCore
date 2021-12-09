using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Yolo5.NetCore.Extensions;
using Yolo5.NetCore.Models;

namespace Yolo5.NetCore
{
    public class Yolo<T> : IDisposable where T : YoloModel
    {
        private readonly InferenceSession _inferenceSession;
        private readonly T _model;

        public Yolo()
        {
            _model = Activator.CreateInstance<T>();
        }

        public Yolo(string model, SessionOptions opts = null) : this()
        {
            var binary = File.ReadAllBytes(model);
            _inferenceSession = new InferenceSession(binary, opts ?? new SessionOptions());
        }

        public Yolo(Stream model, SessionOptions opts = null) : this()
        {
            using var reader = new BinaryReader(model);
            _inferenceSession =
                new InferenceSession(reader.ReadBytes((int)model.Length), opts ?? new SessionOptions());
        }

        public Yolo(byte[] model, SessionOptions opts = null) : this()
        {
            _inferenceSession = new InferenceSession(model, opts ?? new SessionOptions());
        }

        public void Dispose()
        {
            _inferenceSession.Dispose();
        }

        private static float Sigmoid(float value)
        {
            return 1 / (1 + (float) Math.Exp(-value));
        }

        private static float[] ToXyXy(IReadOnlyList<float> source)
        {
            var result = new float[4];

            result[0] = source[0] - source[2] / 2f;
            result[1] = source[1] - source[3] / 2f;
            result[2] = source[0] + source[2] / 2f;
            result[3] = source[1] + source[3] / 2f;

            return result;
        }

        public float Clamp(float value, float min, float max)
        {
            return value < min ? min : value > max ? max : value;
        }

        private Bitmap ResizeImage(Image image)
        {
            var format = image.PixelFormat;
            var output = new Bitmap(_model.Width, _model.Height, format);

            var (w, h) = (image.Width, image.Height);
            var (xRatio, yRatio) = (_model.Width / (float) w, _model.Height / (float) h);    
            var ratio = Math.Min(xRatio, yRatio);      
            var (width, height) = ((int) (w * ratio), (int) (h * ratio));     
            var (x, y) = (_model.Width / 2 - width / 2, _model.Height / 2 - height / 2);      
            var roi = new Rectangle(x, y, width, height);    

            using var graphics = Graphics.FromImage(output);
            graphics.Clear(Color.FromArgb(0, 0, 0, 0));   

            graphics.SmoothingMode = SmoothingMode.None;   
            graphics.InterpolationMode = InterpolationMode.Bilinear;   
            graphics.PixelOffsetMode = PixelOffsetMode.Half;    

            graphics.DrawImage(image, roi);   

            return output;
        }

        private Tensor<float> ExtractPixels(Image image)
        {
            var timer = new Stopwatch();
            timer.Start();

            var bitmap = (Bitmap) image;
            var rectangle = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            var bitmapData = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, bitmap.PixelFormat);
            var bytesPerPixel = Image.GetPixelFormatSize(bitmap.PixelFormat) / 8;

            var tensor = new DenseTensor<float>(new[] {1, 3, _model.Height, _model.Width});

            // TODO: Find a safe way to do this, maybe using PinnedMemory.NetCore
            unsafe
            {
                Parallel.For(0, bitmapData.Height, y =>
                {
                    var row = (byte*) bitmapData.Scan0 + y * bitmapData.Stride;

                    Parallel.For(0, bitmapData.Width, x =>
                    {
                        tensor[0, 0, y, x] = row[x * bytesPerPixel + 2] / 255.0F;  
                        tensor[0, 1, y, x] = row[x * bytesPerPixel + 1] / 255.0F;  
                        tensor[0, 2, y, x] = row[x * bytesPerPixel + 0] / 255.0F;  
                    });
                });

                bitmap.UnlockBits(bitmapData);
            }

            timer.Stop();
            Console.WriteLine($"{timer.ElapsedMilliseconds}ms");

            return tensor;
        }

        private DenseTensor<float>[] Inference(Image image)
        {
            Bitmap resized = null;

            if (image.Width != _model.Width || image.Height != _model.Height)
                resized = ResizeImage(image);        

            var inputs = new List<NamedOnnxValue>      
            {
                NamedOnnxValue.CreateFromTensor("images", ExtractPixels(resized ?? image))
            };

            var result = _inferenceSession.Run(inputs);   
            return _model.Outputs.Select(item => result.First(x => x.Name == item).Value as DenseTensor<float>)
                .ToArray();
        }

        private List<YoloPredictionModel> ParseDetect(Tensor<float> output, Image image)
        {
            var result = new ConcurrentBag<YoloPredictionModel>();

            var (w, h) = (image.Width, image.Height);     
            var (xGain, yGain) = (_model.Width / (float) w, _model.Height / (float) h);    
            var gain = Math.Min(xGain, yGain);      

            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2);    

            Parallel.For(0, (int) output.Length / _model.Dimensions, i =>
            {
                if (output[0, i, 4] <= _model.Confidence) return;     

                Parallel.For(5, _model.Dimensions, j =>
                {
                    output[0, i, j] = output[0, i, j] * output[0, i, 4];      
                });

                Parallel.For(5, _model.Dimensions, k =>
                {
                    if (output[0, i, k] <= _model.MulConfidence) return;     

                    var xMin = (output[0, i, 0] - output[0, i, 2] / 2 - xPad) / gain;      
                    var yMin = (output[0, i, 1] - output[0, i, 3] / 2 - yPad) / gain;      
                    var xMax = (output[0, i, 0] + output[0, i, 2] / 2 - xPad) / gain;      
                    var yMax = (output[0, i, 1] + output[0, i, 3] / 2 - yPad) / gain;      

                    xMin = Clamp(xMin, 0, w - 0);      
                    yMin = Clamp(yMin, 0, h - 0);      
                    xMax = Clamp(xMax, 0, w - 1);      
                    yMax = Clamp(yMax, 0, h - 1);      

                    var label = _model.Labels[k - 5];

                    var prediction = new YoloPredictionModel(label, output[0, i, k])
                    {
                        Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                    };

                    result.Add(prediction);
                });
            });

            return result.ToList();
        }

        private List<YoloPredictionModel> ParseSigmoid(IReadOnlyList<DenseTensor<float>> output, Image image)
        {
            var result = new ConcurrentBag<YoloPredictionModel>();

            var (w, h) = (image.Width, image.Height);     
            var (xGain, yGain) = (_model.Width / (float) w, _model.Height / (float) h);    
            var gain = Math.Min(xGain, yGain);      

            var (xPad, yPad) = ((_model.Width - w * gain) / 2, (_model.Height - h * gain) / 2);    

            Parallel.For(0, output.Count, i =>    
            {
                var shapes = _model.Shapes[i];    

                Parallel.For(0, _model.Anchors[0].Length, a =>   
                {
                    Parallel.For(0, shapes, y =>    
                    {
                        Parallel.For(0, shapes, x =>    
                        {
                            var offset = (shapes * shapes * a + shapes * y + x) * _model.Dimensions;

                            var buffer = output[i].Skip(offset).Take(_model.Dimensions).Select(Sigmoid).ToArray();

                            if (buffer[4] <= _model.Confidence) return;     

                            var scores =
                                buffer.Skip(5).Select(b => b * buffer[4]).ToList();      

                            var mulConfidence = scores.Max();    

                            if (mulConfidence <= _model.MulConfidence) return;     

                            var rawX = (buffer[0] * 2 - 0.5f + x) * _model.Strides[i];     
                            var rawY = (buffer[1] * 2 - 0.5f + y) * _model.Strides[i];     

                            var rawW = (float) Math.Pow(buffer[2] * 2, 2) * _model.Anchors[i][a][0];    
                            var rawH = (float) Math.Pow(buffer[3] * 2, 2) * _model.Anchors[i][a][1];    

                            var xyxy = ToXyXy(new[] {rawX, rawY, rawW, rawH});

                            var xMin = Clamp((xyxy[0] - xPad) / gain, 0, w - 0);    
                            var yMin = Clamp((xyxy[1] - yPad) / gain, 0, h - 0);    
                            var xMax = Clamp((xyxy[2] - xPad) / gain, 0, w - 1);    
                            var yMax = Clamp((xyxy[3] - yPad) / gain, 0, h - 1);    

                            var label = _model.Labels[scores.IndexOf(mulConfidence)];

                            var prediction = new YoloPredictionModel(label, mulConfidence)
                            {
                                Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                            };

                            result.Add(prediction);
                        });
                    });
                });
            });

            return result.ToList();
        }

        private List<YoloPredictionModel> ParseOutput(DenseTensor<float>[] output, Image image)
        {
            if (output == null) throw new ArgumentNullException(nameof(output));
            return _model.UseDetect ? Clean(ParseDetect(output[0], image)) : Clean(ParseSigmoid(output, image));
        }

        private List<YoloPredictionModel> Clean(IReadOnlyCollection<YoloPredictionModel> items)
        {
            var result = new List<YoloPredictionModel>(items);

            foreach (var item in items)    
            {
                var list = result.ToList();
                foreach (var current in list.Where(current => current != item))
                {
                    var (rect1, rect2) = (item.Rectangle, current.Rectangle);

                    var intersection = RectangleF.Intersect(rect1, rect2);

                    var intArea = intersection.Area();   
                    var unionArea = rect1.Area() + rect2.Area() - intArea;   
                    var overlap = intArea / unionArea;   

                    if (overlap >= _model.Overlap)
                        if (item.Score >= current.Score)
                            result.Remove(current);
                }
            }

            return result;
        }

        public List<YoloPredictionModel> Predict(Image image)
        {
            var inference = Inference(image);
            return ParseOutput(inference, image);
        }
    }
}