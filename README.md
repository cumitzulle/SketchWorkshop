# SketchWorkshop
SketchWorkshop: Fine-grained Control over Sketch Generation with Pre-trained Models

The project serves as the code for class project of CS3308 2025Fall. 

The project is dedicated to incorporate the capabilities of the three sketch generation frameworks (CLIPasso, DiffSketcher, and M3S) in precise controllability on stroke count, stroke attributes, and sketch style.

The pipeline workflow of the project is designed as below: 

<img src="./assets/屏幕截图 2026-01-17 141955.png" alt="屏幕截图 2026-01-17 141955" style="zoom: 40%;" />

Usage: 

1. Clone the repo:

```
git clone https://github.com/cumitzulle/SketchWorkshop.git
cd SketchWorkshop
```

2. Create and activate `venv`, virtual environment of Python:

```
python -m venv venv
# for Linux
source venv/bin/activate
# for Windows
venv\Scripts\activate
```

3. Install the libraries

```
pip install -r requirements.txt
```

4. Install diffvg

```
cd diffvg
git submodule update --init --recursive
python setup.py install
cd ../
```

5. Run sketch generation with control parameters (under relative root path)

```
python main.py --prompt <prompt_of_the_sketch_in_string> <other_parameters>
```

The project implements integrated methods for fine-grain control over sketch generation, chiefly supporting the following parameters: 

- `--prompt`: Text prompt for sketch generation, compulsory 
- `--stroke-count`: Number of strokes, disable stroke count control module by default
- `--abstraction`: Abstraction level, 0.7 by default within range 0 to 1
- `--style-reference`: Sketch as style reference, disable style control module by default
- `--style-text`: Text as style reference (among several preset styles), disable style control module by default
- `--stroke-width`: Multiplier for stroke width, 1.0 by default
- `--stroke-opacity`: Opacity for strokes, 1.0 by default within range 0 to 1
- `--width-variation`: Variation factor for stroke width, 0.5 by default within range 0 to 1
- `--guidance-scale`: Guidance scale, 7.5 by default
- `--output-dir`: Directory for sketch output, "./output" by default
- `--verbose`: Enable detailed running information output, false by default
- `--skip-diffvg`: Skip the initialization of diffvg, false by default
