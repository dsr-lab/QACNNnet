"""
Utility methods for creating the files xx_texts_to_translsate (found in the data/translations
directory). These files contains a list of strings in the corresponding language.
These files are used when the 'original' language must be kept.

Note that the resulting files obtained by running these methods have been already saved in
the repository, thus it is not necessary to run this module.
"""

from os.path import join

from preprocessing.augmentation.translation import extract_texts_to_translate, PROJECT_ROOT, read_dataset


def save_texts_to_translate(json_dict, language_code):

    texts_to_translate = extract_texts_to_translate(json_dict)

    # Save the original texts to translate
    file_path = join(PROJECT_ROOT, "data", "translations", "{}_texts_to_translate.txt").format(language_code)
    with open(file_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(texts_to_translate) + '\n')

    print("Data extraction completed!")

    return texts_to_translate


def create_texts_to_translate_files():
    # Create file contaning english original strings
    dataset_path = join(PROJECT_ROOT, 'data', 'training_set.json')
    json_dict = read_dataset(dataset_path)
    save_texts_to_translate(json_dict, 'en')

    # Create file contaning german original (translated) strings
    dataset_path = join(PROJECT_ROOT, 'data', 'translated_dataset_en-de.json')
    json_dict = read_dataset(dataset_path)
    save_texts_to_translate(json_dict, 'de')

    # Create file contaning french original (translated) strings
    dataset_path = join(PROJECT_ROOT, 'data', 'translated_dataset_en-fr.json')
    json_dict = read_dataset(dataset_path)
    save_texts_to_translate(json_dict, 'fr')
