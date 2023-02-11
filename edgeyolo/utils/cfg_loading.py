import yaml
import tabulate
import os


class dictConfig:
    __data_dict: dict = {}

    def __init__(self, src=None):
        if src is None:
            return
        if isinstance(src, str):
            if os.path.isfile(src):
                src = yaml.load(open(src), yaml.Loader)
            else:
                raise FileNotFoundError(f"No such file named '{src}'.")
        elif isinstance(src, dict):
            pass
        elif hasattr(src, "keys"):
            pass
        else:
            raise TypeError(f"Get wrong type of src({type(src)}), src type should be str or dict!")

        for k in src.keys():
            self.__data_dict[k] = src[k]

    def __getattr__(self, item):
        if item in self.__data_dict.keys():
            return self.__data_dict[item]
        else:
            # print(self.__data_dict)
            raise AttributeError(f"No attribute named '{item}'.")

    def __getitem__(self, item):
        return getattr(self, item)

    def __repr__(self):
        return tabulate.tabulate(
            [[k, getattr(self, k)] for k in self.__data_dict.keys()],
            ["Keywords", "Values"],
            "fancy_grid"
        )

    def __setattr__(self, key, value):
        self.__data_dict[key] = value
        # print(self.__data_dict)

    def keys(self):
        return self.__data_dict.keys()


if __name__ == '__main__':
    a = dictConfig()
    a.test = 1
    print(a.keys())
    