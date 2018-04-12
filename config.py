import os

class_name = ['absence', 'cooking', 'dishwashing', 'eating', 'other', 'social_activity', 'vacuum_cleaner',
              'watching_tv', 'working', ]
class_index = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# class_name_dict = {'dishwashing': 0, 'social_activity': 1, 'vacuum_cleaner': 2, 'watching_tv': 3, 'other': 4,
#                    'cooking': 5, 'working': 6, 'absence': 7, 'eating': 8}
class_index2name = {index: name for index, name in zip(class_index, class_name)}
class_name2index = {name: index for name, index in zip(class_name, class_index)}

dev_dir = "DCASE2018-task5-dev"
evl_dir = None
fold_meta_dir = "evaluation_setup"
meta_dir = os.path.join(dev_dir, 'meta.txt')
