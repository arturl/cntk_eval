using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.Storage;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;
using System.Threading.Tasks;
using Windows.UI.Xaml.Media.Imaging;
using Windows.Storage.Streams;
using Windows.Graphics.Imaging;
using ImageRecognitionLib;
using System.Diagnostics;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace ImageRecognitionCS
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        public MainPage()
        {
            this.InitializeComponent();
            var t = Run();
        }

        private async Task Run()
        {
            var sw = Stopwatch.StartNew();
            this.text.Text = "Loading model... ";

            var recognizer = new ImageRecognizer("Assets\\ResNet18_ImageNet_CNTK.model");

            sw.Stop();

            this.text.Text += String.Format("Elapsed time: {0} ms", sw.ElapsedMilliseconds);

            var folder = await Windows.ApplicationModel.Package.Current.InstalledLocation.GetFolderAsync("Assets");

            var files = new String[]
            {
                "broccoli.jpg",
                "cauliflower.jpg",
                "snow-leopard.jpg",
                "timber-wolf.jpg",
                "nile_crocodile_1.jpg",
                "American-alligator.jpg"
            };

            foreach (var fileName in files)
            {
                var file = await folder.GetFileAsync(fileName);
                var fileStream = await file.OpenAsync(Windows.Storage.FileAccessMode.Read);

                var decoder = await BitmapDecoder.CreateAsync(fileStream);

                BitmapTransform transform = new BitmapTransform()
                {
                    ScaledHeight = recognizer.GetRequiredHeight(),
                    ScaledWidth = recognizer.GetRequiredWidth()
                };

                PixelDataProvider pixelData = await decoder.GetPixelDataAsync(
                    BitmapPixelFormat.Rgba8,
                    BitmapAlphaMode.Straight,
                    transform,
                    ExifOrientationMode.RespectExifOrientation,
                    ColorManagementMode.DoNotColorManage);

                var data = pixelData.DetachPixelData();
                sw = Stopwatch.StartNew();

                string objectName = "?";
                var task = Task.Run(() =>
                {
                    try
                    {
                        objectName = recognizer.RecognizeObject(data);
                    }
                    catch
                    {
                        objectName = "error";
                    }
                });

                await task;

                sw.Stop();

                this.text.Text += String.Format("\n{0} -> {1}. Elapsed time: {2} ms", fileName, objectName, sw.ElapsedMilliseconds);

            }

        }
    }
}
