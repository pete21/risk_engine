class ProtoDecoder(object):
    def __init__(self, proto_type):
        self._proto_type = proto_type

    def __call__(self, field_name, data):
        if data is None:
            raise TypeError('Required {} request field missing'.format(
                field_name))
        proto = self._proto_type()
        proto.MergeFromString(data)
        return proto
