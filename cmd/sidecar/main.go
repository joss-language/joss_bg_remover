package main

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

var (
	modelDownloadMutex sync.Mutex
	isDownloadingModel bool
)

type request struct {
	Protocol string        `json:"protocol"`
	ID       string        `json:"id"`
	Method   string        `json:"method"`
	Args     []interface{} `json:"args"`
}

type response struct {
	ID     string      `json:"id"`
	Result interface{} `json:"result,omitempty"`
	Error  interface{} `json:"error,omitempty"`
}

func main() {
	if len(os.Args) >= 4 && os.Args[1] == "remove_background" {
		imagePath := os.Args[2]
		outputPath := os.Args[3]
		res, err := removeBackground(imagePath, outputPath, "")
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		jsonBytes, _ := json.Marshal(res)
		fmt.Println(string(jsonBytes))
		os.Exit(0)
	}

	var req request
	if err := json.NewDecoder(io.LimitReader(os.Stdin, 16<<20)).Decode(&req); err != nil {
		write(response{Error: map[string]string{"code": "BAD_REQUEST", "message": err.Error()}})
		return
	}
	if req.Protocol != "joss-rpc-v1" {
		write(response{ID: req.ID, Error: map[string]string{"code": "BAD_PROTOCOL", "message": "se requiere joss-rpc-v1"}})
		return
	}

	result, err := dispatch(req.Method, req.Args)
	if err != nil {
		write(response{ID: req.ID, Error: map[string]string{"code": "BG_REMOVER_ERROR", "message": err.Error()}})
		return
	}
	write(response{ID: req.ID, Result: result})
}

func dispatch(method string, args []interface{}) (interface{}, error) {
	switch method {
	case "ping":
		return map[string]interface{}{"status": "pong", "plugin": "joss_bg_remover", "engine": "pure_go_onnx"}, nil
	case "preload", "init":
		go ensureONNXModelPathAsync()
		return map[string]interface{}{"status": "preload_started", "plugin": "joss_bg_remover"}, nil
	case "remove_background", "process":
		if len(args) == 0 {
			return nil, fmt.Errorf("se requiere un objeto con image_path u objeto de opciones")
		}
		params, _ := args[0].(map[string]interface{})
		imagePath, _ := params["image_path"].(string)
		outputPath, _ := params["output_path"].(string)
		modelPath, _ := params["model_path"].(string)
		if imagePath == "" {
			return nil, fmt.Errorf("image_path es requerido")
		}
		return removeBackground(imagePath, outputPath, modelPath)
	default:
		return nil, fmt.Errorf("método no soportado: %s", method)
	}
}

func removeBackground(imagePath, outputPath, modelPath string) (map[string]interface{}, error) {
	if outputPath == "" {
		ext := filepath.Ext(imagePath)
		base := strings.TrimSuffix(imagePath, ext)
		outputPath = base + "_no_bg.png"
	}

	// 1. Direct Pure Go ONNX Inference (No HTTP, No Flask, No Python)
	if err := removeBackgroundONNX(imagePath, outputPath, modelPath); err == nil {
		return map[string]interface{}{
			"success":     true,
			"output_path": outputPath,
			"provider":    "pure_go_onnx_native",
		}, nil
	}

	// 2. Pure Go Native Fallback Engine (Color segmentation & Alpha channel)
	if err := removeBackgroundNativeGo(imagePath, outputPath); err == nil {
		return map[string]interface{}{
			"success":     true,
			"output_path": outputPath,
			"provider":    "go_color_segmentation",
		}, nil
	}

	return nil, fmt.Errorf("no se pudo remover el fondo de la imagen")
}

func removeBackgroundONNX(imagePath, outputPath, modelPath string) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("ONNX error de runtime: %v", r)
		}
	}()

	if modelPath == "" {
		modelPath = ensureONNXModelPath()
	}
	if modelPath == "" || !fileExists(modelPath) {
		return fmt.Errorf("modelo ONNX no disponible")
	}

	modelData, err := os.ReadFile(modelPath)
	if err != nil {
		return fmt.Errorf("error leyendo modelo ONNX: %w", err)
	}

	backend := gorgonnx.NewGraph()
	model := onnx.NewModel(backend)
	if err := model.UnmarshalBinary(modelData); err != nil {
		return fmt.Errorf("error decodificando modelo ONNX: %w", err)
	}

	file, err := os.Open(imagePath)
	if err != nil {
		return err
	}
	defer file.Close()

	srcImg, _, err := image.Decode(file)
	if err != nil {
		return fmt.Errorf("error decodificando imagen: %w", err)
	}

	bounds := srcImg.Bounds()
	origW, origH := bounds.Dx(), bounds.Dy()
	if origW == 0 || origH == 0 {
		return fmt.Errorf("dimensiones de imagen inválidas")
	}

	tensorWidth, tensorHeight := 1024, 1024
	inputTensorData := make([]float32, 1*3*tensorHeight*tensorWidth)

	for y := 0; y < tensorHeight; y++ {
		srcY := bounds.Min.Y + (y * origH / tensorHeight)
		for x := 0; x < tensorWidth; x++ {
			srcX := bounds.Min.X + (x * origW / tensorWidth)
			r, g, b, _ := srcImg.At(srcX, srcY).RGBA()
			r32 := (float32(r>>8)/255.0 - 0.5) / 0.5
			g32 := (float32(g>>8)/255.0 - 0.5) / 0.5
			b32 := (float32(b>>8)/255.0 - 0.5) / 0.5

			inputTensorData[0*tensorHeight*tensorWidth+y*tensorWidth+x] = r32
			inputTensorData[1*tensorHeight*tensorWidth+y*tensorWidth+x] = g32
			inputTensorData[2*tensorHeight*tensorWidth+y*tensorWidth+x] = b32
		}
	}

	inputTensor := tensor.New(tensor.WithShape(1, 3, tensorHeight, tensorWidth), tensor.WithBacking(inputTensorData))
	if err := model.SetInput(0, inputTensor); err != nil {
		return fmt.Errorf("error seteando entrada ONNX: %w", err)
	}

	if err := backend.Run(); err != nil {
		return fmt.Errorf("error ejecutando grafo ONNX: %w", err)
	}

	outputs, err := model.GetOutputTensors()
	if err != nil || len(outputs) == 0 {
		return fmt.Errorf("error obteniendo salida ONNX")
	}

	maskData, ok := outputs[0].Data().([]float32)
	if !ok {
		return fmt.Errorf("formato de datos de salida ONNX no válido")
	}

	outImg := image.NewNRGBA(bounds)

	for y := 0; y < origH; y++ {
		maskY := y * tensorHeight / origH
		for x := 0; x < origW; x++ {
			maskX := x * tensorWidth / origW
			idx := maskY*tensorWidth + maskX
			var maskVal float32
			if idx < len(maskData) {
				maskVal = maskData[idx]
			}
			if maskVal < 0 {
				maskVal = 0
			} else if maskVal > 1 {
				maskVal = 1
			}

			alpha := uint8(maskVal * 255.0)
			srcColor := srcImg.At(bounds.Min.X+x, bounds.Min.Y+y)
			r, g, b, _ := srcColor.RGBA()
			outImg.SetNRGBA(bounds.Min.X+x, bounds.Min.Y+y, color.NRGBA{
				R: uint8(r >> 8),
				G: uint8(g >> 8),
				B: uint8(b >> 8),
				A: alpha,
			})
		}
	}

	_ = os.MkdirAll(filepath.Dir(outputPath), 0755)
	outFile, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	return png.Encode(outFile, outImg)
}

func removeBackgroundNativeGo(imagePath, outputPath string) error {
	file, err := os.Open(imagePath)
	if err != nil {
		return err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return fmt.Errorf("error decodificando imagen: %w", err)
	}

	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	if width == 0 || height == 0 {
		return fmt.Errorf("dimensiones de imagen inválidas")
	}

	// Muestreo de las cuatro esquinas y bordes superiores e inferiores para determinar el perfil de fondo
	corners := []color.Color{
		img.At(bounds.Min.X, bounds.Min.Y),
		img.At(bounds.Max.X-1, bounds.Min.Y),
		img.At(bounds.Min.X, bounds.Max.Y-1),
		img.At(bounds.Max.X-1, bounds.Max.Y-1),
		img.At(bounds.Min.X+width/2, bounds.Min.Y),
		img.At(bounds.Min.X+width/4, bounds.Min.Y),
		img.At(bounds.Min.X+3*width/4, bounds.Min.Y),
	}

	var sumR, sumG, sumB uint64
	for _, c := range corners {
		r, g, b, _ := c.RGBA()
		sumR += uint64(r >> 8)
		sumG += uint64(g >> 8)
		sumB += uint64(b >> 8)
	}
	count := uint64(len(corners))
	bgR := float64(sumR / count)
	bgG := float64(sumG / count)
	bgB := float64(sumB / count)

	outImg := image.NewNRGBA(bounds)

	for y := bounds.Min.Y; y < height; y++ {
		for x := bounds.Min.X; x < width; x++ {
			c := img.At(x, y)
			r, g, b, a := c.RGBA()
			r8, g8, b8, a8 := uint8(r>>8), uint8(g>>8), uint8(b>>8), uint8(a>>8)

			dr := float64(r8) - bgR
			dg := float64(g8) - bgG
			db := float64(b8) - bgB
			distance := math.Sqrt(dr*dr + dg*dg + db*db)

			var alpha uint8
			if distance < 45.0 {
				alpha = 0
			} else if distance < 80.0 {
				factor := (distance - 45.0) / 35.0
				alpha = uint8(float64(a8) * factor)
			} else {
				alpha = a8
			}

			// Mantiene exactamente los colores RGB originales de la cara y ropa de la persona
			outImg.SetNRGBA(x, y, color.NRGBA{
				R: r8,
				G: g8,
				B: b8,
				A: alpha,
			})
		}
	}

	_ = os.MkdirAll(filepath.Dir(outputPath), 0755)
	outFile, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	return png.Encode(outFile, outImg)
}

func ensureONNXModelPath() string {
	if path := findONNXModelPath(); path != "" {
		return path
	}

	go ensureONNXModelPathAsync()
	return ""
}

func ensureONNXModelPathAsync() {
	modelDownloadMutex.Lock()
	if isDownloadingModel || findONNXModelPath() != "" {
		modelDownloadMutex.Unlock()
		return
	}
	isDownloadingModel = true
	modelDownloadMutex.Unlock()

	defer func() {
		modelDownloadMutex.Lock()
		isDownloadingModel = false
		modelDownloadMutex.Unlock()
	}()

	cwd, _ := os.Getwd()
	targetDir := filepath.Join(cwd, "plugins", "joss_bg_remover", "models")
	_ = os.MkdirAll(targetDir, 0755)
	targetFile := filepath.Join(targetDir, "model.onnx")

	modelURL := strings.TrimSpace(os.Getenv("ONNX_MODEL_URL"))
	if modelURL == "" {
		modelURL = "https://huggingface.co/briaai/RMBG-1.4/resolve/main/onnx/model.onnx"
	}

	fmt.Fprintf(os.Stderr, "[joss_bg_remover] Descargando modelo ONNX de IA en segundo plano desde %s...\n", modelURL)
	if err := downloadFile(modelURL, targetFile); err == nil {
		fmt.Fprintf(os.Stderr, "[joss_bg_remover] Modelo ONNX listo en %s\n", targetFile)
	} else {
		fmt.Fprintf(os.Stderr, "[joss_bg_remover] Advertencia: Error descargando modelo ONNX en segundo plano: %v\n", err)
	}
}

func downloadFile(url, destination string) error {
	client := &http.Client{
		Timeout: 15 * time.Minute,
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP error %d al descargar modelo", resp.StatusCode)
	}

	tmpFile := destination + ".tmp"
	out, err := os.Create(tmpFile)
	if err != nil {
		return err
	}

	_, err = io.Copy(out, resp.Body)
	_ = out.Close()
	if err != nil {
		_ = os.Remove(tmpFile)
		return err
	}

	return os.Rename(tmpFile, destination)
}

func findONNXModelPath() string {
	if envPath := strings.TrimSpace(os.Getenv("ONNX_MODEL_PATH")); envPath != "" {
		return envPath
	}
	cwd, _ := os.Getwd()
	candidates := []string{
		filepath.Join(cwd, "models", "model.onnx"),
		filepath.Join(cwd, "model.onnx"),
		filepath.Join(cwd, "plugins", "joss_bg_remover", "models", "model.onnx"),
	}
	for _, p := range candidates {
		if fileExists(p) {
			return p
		}
	}
	return ""
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func write(val response) {
	_ = json.NewEncoder(os.Stdout).Encode(val)
}
