import copy
import json
import os
import io
import zipfile
import shutil
from tempfile import NamedTemporaryFile

import magic_pdf.model as model_config
import urllib.parse
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from loguru import logger

import magic_pdf.model as model_config
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import DataWriter, FileBasedDataWriter
from magic_pdf.data.data_reader_writer.s3 import S3DataReader, S3DataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.libs.config_reader import get_bucket_name, get_s3_config
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.operators.models import InferenceResult
from magic_pdf.operators.pipes import PipeResult

model_config.__use_inside_model__ = True

app = FastAPI()

def json_md_dump(
    model_json,
    middle_json,
    md_writer,
    pdf_name,
    content_list,
    md_content,
):
    # Write model results to model.json
    orig_model_list = copy.deepcopy(model_json)
    md_writer.write_string(
        f'{pdf_name}_model.json',
        json.dumps(orig_model_list, ensure_ascii=False, indent=4),
    )

    # Write intermediate results to middle.json
    md_writer.write_string(
        f'{pdf_name}_middle.json',
        json.dumps(middle_json, ensure_ascii=False, indent=4),
    )

    # Write text content results to content_list.json
    md_writer.write_string(
        f'{pdf_name}_content_list.json',
        json.dumps(content_list, ensure_ascii=False, indent=4),
    )

    # Write results to .md file
    md_writer.write_string(
        f'{pdf_name}.md',
        md_content,
    )


@app.post('/pdf_parse', tags=['projects'], summary='Parse PDF file')
async def pdf_parse_main(
    pdf_file: UploadFile = File(...),
    parse_method: str = 'auto',
    model_json_path: str = None,
    is_json_md_dump: bool = True,
):
    """Execute the process of converting PDF to JSON and MD, outputting MD and
    JSON files to the specified directory.

    :param pdf_file: The PDF file to be parsed
    :param parse_method: Parsing method, can be auto, ocr, or txt. Default is auto. If results are not satisfactory, try ocr
    :param model_json_path: Path to existing model data file. If empty, use built-in model. PDF and model_json must correspond
    :param is_json_md_dump: Whether to write parsed data to .json and .md files. Default is True. Different stages of data will be written to different .json files (3 in total), md content will be saved to .md file  # noqa E501
    :param output_dir: Output directory for results. A folder named after the PDF file will be created to store all results
    """
    output_dir: str = "temp_output"
    try:
        # Create a temporary file to store the uploaded PDF
        with NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(await pdf_file.read())
            temp_pdf_path = temp_pdf.name

        pdf_name = ".".join(os.path.basename(pdf_file.filename).split(".")[:-1])

        if output_dir:
            output_path = os.path.join(output_dir, pdf_name)
        else:
            output_path = os.path.join(os.path.dirname(temp_pdf_path), pdf_name)

        output_image_path = os.path.join(output_path, 'images')

        # Get parent path of images for relative path in .md and content_list.json
        image_path_parent = os.path.basename(output_image_path)

        pdf_bytes = open(temp_pdf_path, 'rb').read()  # Read binary data of PDF file

        if model_json_path:
            # Read original JSON data of PDF file parsed by model, list type
            model_json = json.loads(open(model_json_path, 'r', encoding='utf-8').read())
        else:
            model_json = []

        # Execute parsing steps
        image_writer, md_writer = FileBasedDataWriter(
            output_image_path
        ), FileBasedDataWriter(output_path)

        ds = PymuDocDataset(pdf_bytes)
        # Choose parsing method
        if parse_method == 'auto':
            if ds.classify() == SupportedPdfParseMethod.OCR:
                parse_method = 'ocr'
            else:
                parse_method = 'txt'

        if parse_method not in ['txt', 'ocr']:
            logger.error('Unknown parse method, only auto, ocr, txt allowed')
            return JSONResponse(
                content={'error': 'Invalid parse method'}, status_code=400
            )

        if len(model_json) == 0:
            if parse_method == 'ocr':
                infer_result = ds.apply(doc_analyze, ocr=True)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)

        else:
            infer_result = InferenceResult(model_json, ds)

        if len(model_json) == 0 and not model_config.__use_inside_model__:
                logger.error('Need model list input')
                return JSONResponse(
                    content={'error': 'Model list input required'}, status_code=400
                )
        if parse_method == 'ocr':
            pipe_res = infer_result.pipe_ocr_mode(image_writer)
        else:
            pipe_res = infer_result.pipe_txt_mode(image_writer)


        # Save results in text and md format
        content_list = pipe_res.get_content_list(image_path_parent, drop_mode='none')
        md_content = pipe_res.get_markdown(image_path_parent, drop_mode='none')
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(pdf_name + ".md", md_content)
            zip_file.writestr(pdf_name + "_model.json", json.dumps(copy.deepcopy(infer_result._infer_res)))
            zip_file.writestr(pdf_name + "_content_list.json", json.dumps(content_list))
            zip_file.writestr(pdf_name + "_middle.json", json.dumps(pipe_res._pipe_res))
            if(os.path.exists(output_image_path)):
                for file in os.listdir(output_image_path):
                    file_path = os.path.join(output_image_path, file)
                    zip_file.write(file_path, "images/"+file)
                
        # 将 ZIP 文件的指针移到开始位置
        zip_buffer.seek(0)

        if(os.path.exists(output_path)):
            shutil.rmtree(output_path)

        if(pdf_name.isascii()):
            disposition_header = f"attachment; filename={pdf_name}.zip"
        else:
            encoded_pdf_name = urllib.parse.quote(pdf_name)
            disposition_header = f"attachment;filename*=UTF-8''{encoded_pdf_name}.zip"

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": disposition_header}
        )

    except Exception as e:
        logger.exception(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Clean up the temporary file
        if 'temp_pdf_path' in locals():
            os.unlink(temp_pdf_path)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=16588)