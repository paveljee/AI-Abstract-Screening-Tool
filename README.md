# Article Screening Tool

A web-based application for screening academic articles from RIS files using AI-powered analysis. This tool helps researchers automate the article screening process for systematic reviews.

> [!TIP]
> Features specific to [this fork](https://github.com/paveljee/AI-Abstract-Screening-Tool) are formatted as tips or **in bold.**

## Features

- Upload multiple RIS files via drag-and-drop or file selection
- Automated article screening using GPT-4 **or GPT-OSS (via llama.cpp)**
- Duplicate detection and removal
- Real-time progress tracking
- Summary statistics
- CSV export of results
- Interactive web interface

## Prerequisites

### Python Dependencies
```bash
pip install flask
pip install flask-session
pip install pandas
pip install openai
pip install python-dotenv
pip install aiohttp
pip install tenacity
pip install werkzeug
```

> [!TIP]
> Opt in for easier installation using `uv` package manager: [instructions](https://docs.astral.sh/uv/getting-started/installation/).

### Environment Setup
Create a `.env` file in the project root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

> [!TIP]
> Set the API key to a string containing `llama` if you use llama.cpp inference:
> ```
> OPENAI_API_KEY=llama-server
> ```

## Project Structure
```
article-screening-tool/
├── app.py              # Main Flask application
├── .env               # Environment variables
├── uploads/           # Directory for uploaded files
├── templates/         # HTML templates
│   └── index.html    # Main interface template
└── README.md         # This file
```

## Installation

1. Clone the repository or download the files
```bash
git clone <repository-url>
cd article-screening-tool
```

2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

> [!TIP]
> Not needed if using `uv`.

3. Install dependencies
```bash
pip install -r requirements.txt
```

> [!TIP]
> Not needed if using `uv`.

4. Create the uploads directory
```bash
mkdir uploads
```

5. Set up your OpenAI API key in .env file
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

> [!TIP]
> Use `uv run python app.py` if you opted for `uv` installation.

2. Open a web browser and navigate to:
```
http://localhost:5000
```

> [!TIP]
> To enjoy local inference, ensure that your system aligns with minimal requirements by consulting this [guide](https://github.com/ggml-org/llama.cpp/discussions/15396).
>
> Next, install [llama.cpp](https://github.com/ggml-org/llama.cpp/) and download a model file (e.g., [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/)), and run a `llama-server` command.
>
> Consider this sample configuration:
> ```bash
> llama-server \
>   --host 0.0.0.0 \
>   --port 8000 \
>   --model /path/to/gpt-oss-20b-mxfp4.gguf \
>   --alias ggml-org/gpt-oss-20b-GGUF \
>   --n-gpu-layers -1 \
>   --ctx-size 4096 \
>   --batch-size 1 \
>   --threads 1 \
>   --seed 42 \
>   --verbose \
>   --flash-attn on \
>   --jinja \
>   --chat-template-kwargs '{"reasoning_effort": "high"}'
>```

3. Using the interface:
   - Enter your inclusion criteria in the text area
   - Drag and drop RIS files or use the file browser
   - Click "Upload and Process"
   - Monitor progress in real-time
   - Download results as CSV when complete

## RIS File Format

The tool accepts RIS files with the following fields:
- Title fields (TI, T1, CT, BT)
- Abstract fields (AB, N2)
- Author fields (AU, A1, A2, A3, A4)
- Year fields (PY, Y1)

Example RIS format:
```
TY  - JOUR
TI  - Article Title
AU  - Author Name
PY  - 2023
AB  - Abstract text
ER  -
```

## API Endpoints

- `GET /`: Main interface
- `POST /upload`: File upload endpoint
- `GET /process`: Server-sent events for processing status
- `GET /download_csv/<filename>`: Download results

## Processing Pipeline

1. File Upload
   - Validates file format and size
   - Stores files in uploads directory

2. RIS Parsing
   - Extracts metadata from RIS files
   - Identifies duplicates

3. Article Screening
   - Analyzes each article using OpenAI's GPT-4 **or GPT-OSS**
   - Applies inclusion criteria
   - Categorizes as Include/Exclude/Unclear

4. Results Generation
   - Compiles screening results
   - Generates summary statistics
   - Creates downloadable CSV

## Error Handling

The application includes comprehensive error handling for:
- Invalid file formats
- Upload failures
- Processing errors
- API timeouts
- Connection issues

## Security Considerations

- File size limit: 16MB
- Allowed file extensions: .ris only
- Secure filename handling
- Session-based file tracking
- Environment variable protection

## Limitations

- Requires OpenAI API key **or a running llama.cpp server**
- Processing speed depends on API rate limits
- Large files may take significant time to process
- Internet connection required **(for installation only if using llama.cpp)**

## Troubleshooting

1. If files won't upload:
   - Check file extension is .ris
   - Verify file size is under 16MB
   - Ensure uploads directory exists and is writable

2. If processing fails:
   - Check OpenAI API key is valid **or llama-server is properly configured**
   - Verify internet connection
   - Look for errors in console logs

3. If download doesn't work:
   - Ensure processing completed successfully
   - Check browser download permissions
   - Verify file exists in uploads directory

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the GNU General Public License v3.0 with additional restrictions - see the [LICENSE](LICENSE) file for details.

Key Restrictions:
- No commercial use without explicit permission
- Attribution required
- Source code must be made available for any modifications
- Network use restrictions apply

For licensing inquiries, please contact: [Jin Kyu Kim, MD; jjk.kim@mail.utoronto.ca]

**For questions about [this fork](https://github.com/paveljee/AI-Abstract-Screening-Tool), please contact [@paveljee](https://github.com/paveljee)**

## Acknowledgments

- Bootstrap for UI components
- FontAwesome for icons
- jQuery for JavaScript functionality
- OpenAI for GPT-4 API and GPT-OSS
- Georgi Gerganov and the GGML community for llama.cpp and model releases

## Future Improvements

- Batch processing for large files
- Custom screening criteria templates
- Export in multiple formats
- Advanced duplicate detection
- User authentication
- Result persistence
- Screening history

## Contact

[Jin Kyu Kim, MD; jjk.kim@mail.utoronto.ca]
