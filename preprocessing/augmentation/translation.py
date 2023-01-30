import gc
from os.path import exists, join, dirname

import argostranslate.package
import argostranslate.translate

from tqdm import tqdm
import uuid
import json
import torch

################################################################################
# CONSTANTS
################################################################################
PROJECT_ROOT = dirname(dirname(dirname(__file__)))

USE_TRANSLATED_QUERY = True
USE_TRANSLATED_CONTEXT = True
USE_TRANSLATED_ANSWER = True

TRANSLATE_DEBUG = False


def read_dataset(dataset_path):
    """
    Read the dataset from the file system
    :param dataset_path: (str) the path where the dataset is stored
    :return: the read dataset.
    """

    with open(dataset_path, "r") as file:
        print(f'opening file: {dataset_path}')
        data = json.loads(file.read())

    return data


def create_translated_dataset(json_dict,
                              translated_texts,
                              original_texts,
                              output_file_path,
                              use_translated_query=USE_TRANSLATED_QUERY,
                              use_translated_context=USE_TRANSLATED_CONTEXT,
                              use_translated_answer=USE_TRANSLATED_ANSWER):
    """
    Create a new version of the dataset based on some parameters that are used
    for deciding which part must be translated.
    :param json_dict: (dict) the json to translate
    :param translated_texts: (list) a list containing all the previously translated texts
    :param original_texts: (list) a list containing the texts in the original language
    :param output_file_path: (str) the path where to save the new dataset
    :param use_translated_query: (bool) if True, then the translated version of the query is used
    :param use_translated_context: (bool) if True, then the translated version of the context is used
    :param use_translated_answer: (bool) if True, then the translated version of the answer is used
    :return:
    """
    print("Dataset translation started...")

    if TRANSLATE_DEBUG:
        data = json_dict["data"][:1]
    else:
        data = json_dict["data"]

    text_idx = 0

    for element in tqdm(data):
        paragraphs = element["paragraphs"]

        for paragraph in paragraphs:
            if use_translated_context:
                paragraph["context"] = translated_texts[text_idx]
            else:
                paragraph["context"] = original_texts[text_idx]
            text_idx += 1

            for qas in paragraph["qas"]:
                qas["id"] = str(uuid.uuid1())
                if use_translated_query:
                    qas["question"] = translated_texts[text_idx]
                else:
                    qas["question"] = original_texts[text_idx]
                text_idx += 1

                for answer in qas["answers"]:
                    if use_translated_answer:
                        answer["text"] = translated_texts[text_idx]
                    else:
                        answer["text"] = original_texts[text_idx]
                    text_idx += 1

    print("Dataset translation completed!")

    with open(output_file_path, 'w', encoding='utf8') as file:
        # json.dump(json_dict, file, ensure_ascii=False, indent=4)
        file.write(json.dumps(json_dict, ensure_ascii=False))


def extract_texts_to_translate(json_dict):
    """
    Read a dictionary representing a dataset with the same attributes of the SQuAD
    dataset. Each read text is then stored in a list, and eventually saved into a txt
    file. The latter will be used from other methods for actually translating a dataset.
    :param json_dict: (dict) the dataset to read
    :return: (list) the translated texts
    """
    print("Data extraction started...")

    if TRANSLATE_DEBUG:
        data = json_dict["data"][:1]
    else:
        data = json_dict["data"]

    texts_to_translate = []

    for element in tqdm(data):
        paragraphs = element["paragraphs"]

        for paragraph in paragraphs:
            context = paragraph["context"]

            texts_to_translate.append(context.replace('\n', ''))

            for qas in paragraph["qas"]:
                question = qas["question"]
                texts_to_translate.append(question.replace('\n', ''))

                for answer in qas["answers"]:
                    answer_text = answer["text"]

                    texts_to_translate.append(answer_text.replace('\n', ''))

    print("Data extraction completed!")

    return texts_to_translate


def retrieve_original_texts(original_code):
    """
    Read the full list of texts contained in a dataset
    :param original_code: (str) the language of the texts
    :return: (list) a list of texts
    """
    original_data_list = []

    file_path = join(PROJECT_ROOT, "data", "translations", "{}_texts_to_translate.txt").format(original_code)

    if exists(file_path):
        with open(file_path, 'r') as f:
            original_data = f.read()
            original_data_list = original_data.split('\n')[:-1]

    return original_data_list


def translate(from_code, to_code, texts_to_translate):
    """
    Wrapper method that translate a list of texts either by using argos translate, or by
    reading it from a file (if present)
    :param from_code: (str) initial language
    :param to_code: (str) target language
    :param texts_to_translate: (list) list of texts to translate
    :return: (list) list of translated texts
    """
    translated_texts = []

    # Open possibly previously saved files so as to avoid to translate the same sentences
    file_path = join(PROJECT_ROOT, "data", "translations", "translated_texts_{}-{}.txt").format(from_code, to_code)
    if exists(file_path):
        with open(file_path, 'r') as f:
            translated_data = f.read()
            translated_data_list = translated_data.split('\n')[:-1]  # The last line is always false

            texts_to_translate = texts_to_translate[len(translated_data_list):]
            translated_texts += translated_data_list

    # Translate the remaining texts
    with open(file_path, 'a', encoding='utf8') as f:
        print(f'Translation started: {len(texts_to_translate)} texts to translate...')
        for idx, curr_text in enumerate(tqdm(texts_to_translate)):
            # curr_text = curr_text.replace('\n', '')
            new_text = argostranslate.translate.translate(curr_text, from_code, to_code)
            translated_texts.append(new_text)

            if (idx > 0 and idx % 1000 == 0) or idx == len(texts_to_translate) - 1:
                print(f'{idx} saving translations to temporary file...')
                f.write('\n'.join(translated_texts) + '\n')
                translated_texts.clear()

            if idx > 0 and idx % 200 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    # Create a list containing the translated texts
    translated_data_list = []

    if exists(file_path):
        print(f'Translation completed. Now saving new file...')
        with open(file_path, 'r') as f:
            translated_data = f.read()
            translated_data_list = translated_data.split('\n')[:-1]  # The last line is always false

    return translated_data_list


def argotranslate_setup(from_code, to_code):
    """
    Install the required files for working with argos translate.
    :param from_code: (str) initial language
    :param to_code: (str) target language
    """
    # Download and install Argos Translate package
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())


def translate_dataset(dataset_path,
                      from_code='en',
                      to_code='fr',
                      original_code='en',
                      output_file_path=None,
                      use_translated_query=USE_TRANSLATED_QUERY,
                      use_translated_context=USE_TRANSLATED_CONTEXT,
                      use_translated_answer=USE_TRANSLATED_ANSWER):
    """
    Wrapper method for translating a dataset that is in the same form of the SQuAD dataset.
    :param dataset_path: (str) the path where the dataset is stored
    :param from_code: (str) initial language
    :param to_code: (str) target language
    :param original_code: (str) typically set to "en" value, and used for possibly using the texts that are in
        the original dataset
    :param output_file_path: (str) the path where to store the new dataset
    :param use_translated_query: (bool) if True, then the translated version of the query is used
    :param use_translated_context: (bool) if True, then the translated version of the context is used
    :param use_translated_answer: (bool) if True, then the translated version of the answer is used
    :return:
    """
    # Model setup
    argotranslate_setup(from_code, to_code)

    # Read data and create the model translation inputs
    json_dict = read_dataset(dataset_path)
    texts_to_translate = extract_texts_to_translate(json_dict)

    # Retrieve translated and original texts from files (saved in the previous methods)
    translated_texts = translate(from_code, to_code, texts_to_translate)
    original_texts = retrieve_original_texts(original_code)

    assert len(translated_texts) == len(original_texts) and len(original_texts) > 0

    # Finally create the dataset
    create_translated_dataset(json_dict,
                              translated_texts,
                              original_texts,
                              output_file_path=output_file_path,
                              use_translated_query=use_translated_query,
                              use_translated_context=use_translated_context,
                              use_translated_answer=use_translated_answer)

