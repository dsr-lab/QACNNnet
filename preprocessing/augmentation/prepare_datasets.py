import json
from os.path import join

from preprocessing.augmentation.translation import translate_dataset, PROJECT_ROOT, read_dataset

"""
This module contains some convenient methods used for creating the dataset required
for running each test (refer to the docs folder).

Note that each required dataset has been already created and saved in the repository, 
therefore this module should be used only for possibly creating further datasets.
"""

################################################################################
# UTILITY METHODS FOR TRANSLATING THE ORIGINAL ENGLISH TRAINING DATASET INTO
# ANOTHER TARGET LANGUAGE.
################################################################################


def translate_from_en_to_fr():
    """
    Method used for creating the original dataset translated from english to french
    """

    file_to_translate_name = 'training_set.json'
    file_to_translate_path = join(PROJECT_ROOT, 'data', file_to_translate_name)

    output_file_name = 'translated_dataset_en-fr.json'
    output_file_path = join(PROJECT_ROOT, 'data', output_file_name)

    translate_dataset(
        dataset_path=file_to_translate_path,
        from_code='en',
        to_code='fr',
        output_file_path=output_file_path,
        use_translated_query=True,
        use_translated_context=True,
        use_translated_answer=True
    )


def translate_from_en_to_de():
    """
    Method used for creating the original dataset translated from english to german
    """
    file_to_translate_name = 'training_set.json'
    file_to_translate_path = join(PROJECT_ROOT, 'data', file_to_translate_name)

    output_file_name = 'translated_dataset_en-de.json'
    output_file_path = join(PROJECT_ROOT, 'data', output_file_name)

    translate_dataset(
        dataset_path=file_to_translate_path,
        from_code='en',
        to_code='de',
        original_code='en',
        output_file_path=output_file_path,
        use_translated_query=True,
        use_translated_context=True,
        use_translated_answer=True
    )


def backtranslate_from_de_to_en():
    """
    Method used for creating the original dataset translated from english to german
    """
    file_to_translate_name = 'translated_dataset_en-de.json'
    file_to_translate_path = join(PROJECT_ROOT, 'data', file_to_translate_name)

    output_file_name = 'backtranslated_dataset_de-en.json'
    output_file_path = join(PROJECT_ROOT, 'data', output_file_name)

    translate_dataset(
        dataset_path=file_to_translate_path,
        from_code='de',
        to_code='en',
        original_code='en',
        output_file_path=output_file_path,
        use_translated_query=False,
        use_translated_context=True,
        use_translated_answer=True
    )


def translate_from_en_to_es():
    """
    Method used for creating the original dataset translated from english to spanish
    """
    file_to_translate_name = 'training_set.json'
    file_to_translate_path = join(PROJECT_ROOT, 'data', file_to_translate_name)

    output_file_name = 'translated_dataset_en-es.json'
    output_file_path = join(PROJECT_ROOT, 'data', output_file_name)

    translate_dataset(
        dataset_path=file_to_translate_path,
        from_code='en',
        to_code='es',
        original_code='en',
        output_file_path=output_file_path,
        use_translated_query=True,
        use_translated_context=True,
        use_translated_answer=True
    )


################################################################################
# UTILITY METHODS CREATING THE AUGMENTED DATASETS
################################################################################


def testA():
    """
    Method used for creating the dataset required for the testA (see docs folder)
    """

    file_to_translsate_name = 'translated_dataset_fr-en.json'
    new_file_name = 'testA3_en_fr_backtranslateON_queryON_context_OFF_answer_OFF.json'
    original_file_name = 'training_set.json'

    create_augmented_dataset(file_to_translsate_name,
                             new_file_name,
                             original_file_name,
                             from_code='fr',
                             to_code='en',
                             original_code='en',
                             use_translated_query=True,
                             use_translated_context=False,
                             use_translated_answer=False)


def testB():
    """
    Method used for creating the dataset required for the testB (see docs folder)
    """

    file_to_translsate_name = 'translated_dataset_fr-en.json'
    new_file_name = 'testB_en_fr_backtranslateON_queryOFF_context_ON_answer_ON.json'
    original_file_name = 'training_set.json'

    create_augmented_dataset(file_to_translsate_name,
                             new_file_name,
                             original_file_name,
                             from_code='fr',
                             to_code='en',
                             original_code='en',
                             use_translated_query=False,
                             use_translated_context=True,
                             use_translated_answer=True)


def testC():
    """
    Method used for creating the dataset required for the testC (see docs folder)
    """

    file_to_translsate_name = 'translated_dataset_fr-en.json'
    new_file_name = 'testC_en_fr_backtranslateON_queryON_context_ON_answer_ON.json'
    original_file_name = 'training_set.json'

    create_augmented_dataset(file_to_translsate_name,
                             new_file_name,
                             original_file_name,
                             from_code='fr',
                             to_code='en',
                             original_code='en',
                             use_translated_query=True,
                             use_translated_context=True,
                             use_translated_answer=True)


def testD():
    """
    Method used for creating the dataset required for the testD (see docs folder).
    """

    file_to_translsate_name = 'translated_dataset_en-fr.json'
    new_file_name = 'testD_en_fr_backtranslateOFF_queryOFF_context_ON_answer_ON.json'
    original_file_name = 'training_set.json'

    create_augmented_dataset(file_to_translsate_name,
                             new_file_name,
                             original_file_name,
                             from_code='en',
                             to_code='fr',
                             original_code='en',
                             use_translated_query=False,
                             use_translated_context=True,
                             use_translated_answer=True)


def testF():
    """
    Method used for creating the dataset required for the testF (see docs folder).
    This method relies on the usage of other previously created datasets, and merge
    them together in a single one.
    """

    # Read the english dataset
    english_file_name = 'training_set.json'
    english_file_path = join(PROJECT_ROOT, 'data', english_file_name)
    english_dict = read_dataset(english_file_path)

    # Read the (partially translated) french dataset
    second_language_file_name = 'testD_en_fr_backtranslateOFF_queryOFF_context_ON_answer_ON.json'
    second_language_file_path = join(PROJECT_ROOT, 'data', second_language_file_name)
    second_language_dict = read_dataset(second_language_file_path)

    # Read the (partially translated) german dataset
    third_language_file_name = 'testE_en_de_backtranslateOFF_queryOFF_context_ON_answer_ON.json'
    third_language_file_path = join(PROJECT_ROOT, 'data', third_language_file_name)
    third_language_dict = read_dataset(third_language_file_path)

    # Take the data attribute from each dictionary
    english_data = english_dict['data']
    second_language_data = second_language_dict['data']
    third_language_data = third_language_dict['data']

    # Extract only relevant part from the translated datasets
    only_second_language_data = second_language_data[len(english_data):]
    only_third_language_data = third_language_data[len(english_data):]

    # Merge all together and ceate the new file
    final_data = english_data + only_second_language_data + only_third_language_data
    new_json_dic = {
        'version': english_dict['version'],
        'data': final_data
    }

    output_file_name = 'testF_en_fr_de_backtranslateOFF_queryOFF_context_ON_answer_ON.json'
    output_file_path = join(PROJECT_ROOT, 'data', output_file_name)
    with open(output_file_path, 'w', encoding='utf8') as file:
        file.write(json.dumps(new_json_dic, ensure_ascii=False))


def create_augmented_dataset(file_to_translate_name,
                             new_file_name,
                             original_file_name,
                             from_code,
                             to_code,
                             original_code,
                             use_translated_query,
                             use_translated_context,
                             use_translated_answer):
    """
    Method responsible of creating the new dataset that will be used for testing the efficacy of
    the applied augmentation technique.
    :param file_to_translate_name: (str) The json file that must be translate.
    :param new_file_name: (str) The name of the new json file that this method will create.
    :param original_file_name: (str) The name of the json contaning the original dataset that
        will be appended to the one created by this method.
    :param from_code: (str) starting language code.
    :param to_code: (str) ending language code.
    :param original_code: (str) the language of those strings that must not be translated.
    :param use_translated_query: (bool) enable/disable query translation
    :param use_translated_context: (bool) enable/disable context translation
    :param use_translated_answer: (bool) enable/disable answer translation
    :return:
    """

    file_to_translate_path = join(PROJECT_ROOT, 'data', file_to_translate_name)

    original_file_path = join(PROJECT_ROOT, 'data', original_file_name)

    new_file_path = join(PROJECT_ROOT, 'data', new_file_name)

    translate_dataset(
        dataset_path=file_to_translate_path,
        from_code=from_code,
        to_code=to_code,
        original_code=original_code,
        output_file_path=join(PROJECT_ROOT, 'data', new_file_path),
        use_translated_query=use_translated_query,
        use_translated_context=use_translated_context,
        use_translated_answer=use_translated_answer
    )

    ok_counter = 0
    ko_counter = 0

    with open(new_file_path, "r") as file:
        print(f'opening file: {new_file_path}')
        new_json_dic = json.loads(file.read())

    data = new_json_dic['data']

    for element_idx, element in enumerate(data):
        paragraphs = element["paragraphs"]

        for paragraph_idx in reversed(range(len(paragraphs))):
            curr_paragraph = paragraphs[paragraph_idx]

            context_text = curr_paragraph["context"]
            if use_translated_context:
                curr_paragraph["language"] = to_code
            else:
                curr_paragraph["language"] = original_code

            qas = curr_paragraph["qas"]

            for qas_idx in reversed(range(len(qas))):
                curr_qas = qas[qas_idx]

                question_text = curr_qas["question"]
                if use_translated_query:
                    curr_qas["language"] = to_code
                else:
                    curr_qas["language"] = original_code

                answers = curr_qas["answers"]

                for answer_idx in reversed(range(len(answers))):
                    curr_answer = answers[answer_idx]
                    answer_text = curr_answer["text"]
                    if use_translated_answer:
                        curr_answer['language'] = to_code
                    else:
                        curr_answer['language'] = original_code

                    answer_start = curr_answer["answer_start"]

                    # Update indices only if required
                    if use_translated_context or use_translated_answer:

                        ##############################
                        # Number translation fix
                        ##############################
                        # Number such as 1920,27 are always translate by removing the comma and adding a space
                        # This fix has been applied for avoiding to possibly lose good records
                        answer_text_new = None
                        if len(answer_text.split(' ')) == 2:
                            if answer_text.split(' ')[0].isnumeric() and answer_text.split(' ')[1].isnumeric():
                                answer_text_new = f'{answer_text.split(" ")[0]},{answer_text.split(" ")[1]}'

                        # Check if answer is in the context, otherwise remove the row
                        if answer_text.lower() in context_text.lower():
                            ok_counter += 1
                            curr_answer["answer_start"] = context_text.lower().find(answer_text.lower())
                        elif answer_text_new is not None and answer_text_new.lower() in context_text.lower():
                            ok_counter += 1
                            curr_answer["answer_start"] = context_text.lower().find(answer_text_new.lower())
                        else:
                            ko_counter += 1

                            del answers[answer_idx]  # Delete the answer

                            if len(answers) == 0:  # Check if there exist more answers
                                del qas[qas_idx]  # Delete if no more answers are available

                            if len(qas) == 0:  # Check if there exist more elements in the qas array
                                del paragraphs[paragraph_idx]  # Delete if empty
                    else:
                        ok_counter += 1

    print(f'{ok_counter} elements added\n{ko_counter} elements removed')

    new_json_dic['data'] = data
    new_data = new_json_dic['data']

    original_json_dic = read_dataset(original_file_path)
    original_data = original_json_dic['data']

    final_data = original_data + new_data
    new_json_dic['data'] = final_data

    with open(new_file_path, 'w', encoding='utf8') as file:
        # json.dump(json_dict, file, ensure_ascii=False, indent=4)
        file.write(json.dumps(new_json_dic, ensure_ascii=False))

    return new_json_dic, original_json_dic
