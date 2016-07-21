import ConfigParser

section_names = 'data', 'nmt-architecture'

class Configuration(object):
    def __init__(self, file_name):
        configParser = ConfigParser.SafeConfigParser()
        configParser.optionxform = str  # make option names case sensitive
        found = configParser.read(file_name)
        if not found:
            raise ValueError('No config file found!')
        for name in section_names:
            self.__dict__.update(configParser.items(name))
        #for name in section_names:
        #    for k,v in configParser.items(name):
        #        print k,v,type(v)