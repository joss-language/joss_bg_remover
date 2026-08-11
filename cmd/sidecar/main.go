package main

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
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

func removeBackgroundONNX(imagePath, outputPath, modelPath string) error {
	if modelPath == "" {
		modelPath = findONNXModelPath()
	}
	if modelPath == "" || !fileExists(modelPath) {
		return fmt.Errorf("modelo ONNX no encontrado en %s", modelPath)
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

	c1 := img.At(bounds.Min.X, bounds.Min.Y)
	c2 := img.At(bounds.Max.X-1, bounds.Min.Y)
	c3 := img.At(bounds.Min.X, bounds.Max.Y-1)
	c4 := img.At(bounds.Max.X-1, bounds.Max.Y-1)

	r1, g1, b1, _ := c1.RGBA()
	r2, g2, b2, _ := c2.RGBA()
	r3, g3, b3, _ := c3.RGBA()
	r4, g4, b4, _ := c4.RGBA()

	bgR := uint8((r1 + r2 + r3 + r4) / 4 >> 8)
	bgG := uint8((g1 + g2 + g3 + g4) / 4 >> 8)
	bgB := uint8((b1 + b2 + b3 + b4) / 4 >> 8)

	outImg := image.NewNRGBA(bounds)

	for y := bounds.Min.Y; y < height; y++ {
		for x := bounds.Min.X; x < width; x++ {
			c := img.At(x, y)
			r, g, b, a := c.RGBA()
			r8, g8, b8, a8 := uint8(r>>8), uint8(g>>8), uint8(b>>8), uint8(a>>8)

			dr := float64(r8) - float64(bgR)
			dg := float64(g8) - float64(bgG)
			db := float64(b8) - float64(bgB)
			distance := math.Sqrt(dr*dr + dg*dg + db*db)

			var alpha uint8
			if distance < 35 {
				alpha = 0
			} else if distance < 75 {
				factor := (distance - 35) / 40.0
				alpha = uint8(float64(a8) * factor)
			} else {
				alpha = a8
			}

			outImg.SetNRGBA(x, y, color.NRGBA{R: r8, G: g8, B: b8, A: alpha})
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
