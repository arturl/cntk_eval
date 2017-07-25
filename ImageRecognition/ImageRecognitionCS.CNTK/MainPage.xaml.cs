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
using Windows.Storage.Pickers;

// The Blank Page item template is documented at https://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace ImageRecognitionCS
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        CNTKImageRecognizer cntkRecognizer;

        public MainPage()
        {
            this.InitializeComponent();
            this.cntkPickButton.IsEnabled = false;
            var t = Run();
        }

        private async Task Run()
        {
            var sw = Stopwatch.StartNew();
            this.text.Text = "Loading CNTK Model... ";

            try
            {
                this.cntkRecognizer = await CNTKImageRecognizer.Create("Assets\\ResNet18_ImageNet_CNTK.model", "Assets\\imagenet1000_clsid.txt");
                sw.Stop();
                this.text.Text += $"Elapsed time: {sw.ElapsedMilliseconds} ms";
                this.cntkPickButton.IsEnabled = true;
            }
            catch (Exception ex)
            {
                this.text.Text += $"error: {ex.Message}";
                sw.Stop();
            }

#if false // not using image picker
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
                await RecognizeFile(MLFramework.CNTK, file);
            }
#endif
        }

        private async Task RecognizeFile(StorageFile file)
        {
            var fileStream = await file.OpenAsync(Windows.Storage.FileAccessMode.Read);

            var decoder = await BitmapDecoder.CreateAsync(fileStream);

            uint sHeight = cntkRecognizer.GetRequiredHeight();
            uint sWidth = cntkRecognizer.GetRequiredWidth();

            BitmapTransform transform = new BitmapTransform()
            {
                ScaledHeight = sHeight,
                ScaledWidth = sWidth
            };


            PixelDataProvider pixelData = await decoder.GetPixelDataAsync(
                BitmapPixelFormat.Rgba8,
                BitmapAlphaMode.Straight,
                transform,
                ExifOrientationMode.RespectExifOrientation,
                ColorManagementMode.DoNotColorManage);

            var data = pixelData.DetachPixelData();
            var sw = Stopwatch.StartNew();

            string objectName = "?";
            try
            {
                objectName = await cntkRecognizer.RecognizeObjectAsync(data);
            }
            catch
            {
                objectName = "error";
            }

            sw.Stop();

            this.text.Text += String.Format("\n{0} -> {1}. Elapsed time: {2} ms", file.Name, objectName, sw.ElapsedMilliseconds);
        }

        private async Task GenericImagePicker()
        {
            var picker = new FileOpenPicker();
            picker.ViewMode = PickerViewMode.Thumbnail;
            picker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
            picker.FileTypeFilter.Add(".jpg");
            var file = await picker.PickSingleFileAsync();
            if (file != null)
            {
                await RecognizeFile(file);
            }
        }

        private async void CNTKButton_Click(object sender, RoutedEventArgs e)
        {
            await GenericImagePicker();
        }
    }
}
