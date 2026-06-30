"""Minimal MessagePack decoder for metadata storage.

Supports: nil, bool, int, float, string, map, array.
Used to replace JSON strings for faster filtered search.
"""


# MessagePack format constants
comptime MSGPACK_NIL: UInt8 = 0xC0
comptime MSGPACK_FALSE: UInt8 = 0xC2
comptime MSGPACK_TRUE: UInt8 = 0xC3
comptime MSGPACK_FLOAT64: UInt8 = 0xCB
comptime MSGPACK_UINT8: UInt8 = 0xCC
comptime MSGPACK_UINT16: UInt8 = 0xCD
comptime MSGPACK_UINT32: UInt8 = 0xCE
comptime MSGPACK_UINT64: UInt8 = 0xCF
comptime MSGPACK_INT8: UInt8 = 0xD0
comptime MSGPACK_INT16: UInt8 = 0xD1
comptime MSGPACK_INT32: UInt8 = 0xD2
comptime MSGPACK_INT64: UInt8 = 0xD3
comptime MSGPACK_STR8: UInt8 = 0xD9
comptime MSGPACK_STR16: UInt8 = 0xDA
comptime MSGPACK_STR32: UInt8 = 0xDB
comptime MSGPACK_ARRAY16: UInt8 = 0xDC
comptime MSGPACK_ARRAY32: UInt8 = 0xDD
comptime MSGPACK_MAP16: UInt8 = 0xDE
comptime MSGPACK_MAP32: UInt8 = 0xDF


struct MsgPackValue(Copyable, Movable):
    """A MessagePack value that can be compared against filter predicates."""
    var type: UInt8  # 0=nil, 1=bool, 2=int, 3=float, 4=string, 5=map, 6=array
    var bool_val: Bool
    var int_val: Int64
    var float_val: Float64
    var str_val: String
    var map_keys: List[String]
    var map_values: List[Self]
    var array_values: List[Self]

    def __init__(out self):
        self.type = 0
        self.bool_val = False
        self.int_val = 0
        self.float_val = 0.0
        self.str_val = ""
        self.map_keys = List[String]()
        self.map_values = List[Self]()
        self.array_values = List[Self]()

    def __copyinit__(out self, other: Self):
        self.type = other.type
        self.bool_val = other.bool_val
        self.int_val = other.int_val
        self.float_val = other.float_val
        self.str_val = other.str_val
        self.map_keys = other.map_keys
        self.map_values = other.map_values
        self.array_values = other.array_values

    def __moveinit__(out self, owned other: Self):
        self.type = other.type
        self.bool_val = other.bool_val
        self.int_val = other.int_val
        self.float_val = other.float_val
        self.str_val = other.str_val^
        self.map_keys = other.map_keys^
        self.map_values = other.map_values^
        self.array_values = other.array_values^

    fn is_nil(self) -> Bool:
        return self.type == 0

    fn is_bool(self) -> Bool:
        return self.type == 1

    fn is_int(self) -> Bool:
        return self.type == 2

    fn is_float(self) -> Bool:
        return self.type == 3

    fn is_string(self) -> Bool:
        return self.type == 4

    fn is_map(self) -> Bool:
        return self.type == 5

    fn is_array(self) -> Bool:
        return self.type == 6

    fn as_number(self) -> Float64:
        """Get value as Float64 (works for int and float)."""
        if self.type == 2:
            return Float64(self.int_val)
        elif self.type == 3:
            return self.float_val
        return 0.0

    fn contains_key(self, key: String) -> Bool:
        """Check if map contains key."""
        if self.type != 5:
            return False
        for i in range(len(self.map_keys)):
            if self.map_keys[i] == key:
                return True
        return False

    fn get(self, key: String) -> Self:
        """Get value by key from map."""
        if self.type != 5:
            return MsgPackValue()
        for i in range(len(self.map_keys)):
            if self.map_keys[i] == key:
                return self.map_values[i]
        return MsgPackValue()

    fn contains_value(self, val: Self) -> Bool:
        """Check if array contains value."""
        if self.type != 6:
            return False
        for i in range(len(self.array_values)):
            if self.array_values[i].type == val.type:
                if val.type == 2 and self.array_values[i].int_val == val.int_val:
                    return True
                elif val.type == 3 and self.array_values[i].float_val == val.float_val:
                    return True
                elif val.type == 4 and self.array_values[i].str_val == val.str_val:
                    return True
        return False


fn msgpack_decode(data: UnsafePointer[UInt8, origin_of(data)], length: Int) raises -> MsgPackValue:
    """Decode MessagePack bytes into a MsgPackValue."""
    var pos = 0

    fn read_byte() raises -> UInt8:
        if pos >= length:
            raise Error("MsgPack: unexpected end of data")
        var b = data[pos]
        pos += 1
        return b

    fn read_bytes(n: Int) raises -> UnsafePointer[UInt8, origin_of(data)]:
        if pos + n > length:
            raise Error("MsgPack: unexpected end of data")
        var ptr = data + pos
        pos += n
        return ptr

    fn decode_value() raises -> MsgPackValue:
        var b = read_byte()

        # Positive fixint (0x00 - 0x7f)
        if b < 0x80:
            var val = MsgPackValue()
            val.type = 2
            val.int_val = Int64(b)
            return val

        # Negative fixint (0xe0 - 0xff)
        if b >= 0xe0:
            var val = MsgPackValue()
            val.type = 2
            val.int_val = Int64(Int8(b))
            return val

        # fixmap (0x80 - 0x8f)
        if b >= 0x80 and b < 0x90:
            var count = Int(b & 0x0f)
            var val = MsgPackValue()
            val.type = 5
            val.map_keys = List[String]()
            val.map_values = List[MsgPackValue]()
            for _ in range(count):
                var key = decode_value()
                var value = decode_value()
                val.map_keys.append(key.str_val)
                val.map_values.append(value)
            return val

        # fixarray (0x90 - 0x9f)
        if b >= 0x90 and b < 0xa0:
            var count = Int(b & 0x0f)
            var val = MsgPackValue()
            val.type = 6
            val.array_values = List[MsgPackValue]()
            for _ in range(count):
                val.array_values.append(decode_value())
            return val

        # fixstr (0xa0 - 0xbf)
        if b >= 0xa0 and b < 0xc0:
            var str_len = Int(b & 0x1f)
            var ptr = read_bytes(str_len)
            var val = MsgPackValue()
            val.type = 4
            val.str_val = String(ptr, str_len)
            return val

        # nil
        if b == MSGPACK_NIL:
            return MsgPackValue()

        # true/false
        if b == MSGPACK_TRUE:
            var val = MsgPackValue()
            val.type = 1
            val.bool_val = True
            return val
        if b == MSGPACK_FALSE:
            var val = MsgPackValue()
            val.type = 1
            val.bool_val = False
            return val

        # float64
        if b == MSGPACK_FLOAT64:
            var ptr = read_bytes(8)
            # Big-endian to host endian
            var result: UInt64 = 0
            for i in range(8):
                result = (result << 8) | UInt64(ptr[i])
            var val = MsgPackValue()
            val.type = 3
            val.float_val = bitcast[DType.float64](result)
            return val

        # uint8/16/32/64
        if b == MSGPACK_UINT8:
            var val = MsgPackValue()
            val.type = 2
            val.int_val = Int64(read_byte())
            return val
        if b == MSGPACK_UINT16:
            var ptr = read_bytes(2)
            var val = MsgPackValue()
            val.type = 2
            val.int_val = Int64((UInt16(ptr[0]) << 8) | UInt16(ptr[1]))
            return val
        if b == MSGPACK_UINT32:
            var ptr = read_bytes(4)
            var val = MsgPackValue()
            val.type = 2
            val.int_val = Int64((UInt32(ptr[0]) << 24) | (UInt32(ptr[1]) << 16) | (UInt32(ptr[2]) << 8) | UInt32(ptr[3]))
            return val
        if b == MSGPACK_UINT64:
            var ptr = read_bytes(8)
            var val = MsgPackValue()
            val.type = 2
            var result: UInt64 = 0
            for i in range(8):
                result = (result << 8) | UInt64(ptr[i])
            val.int_val = Int64(result)
            return val

        # int8/16/32/64
        if b == MSGPACK_INT8:
            var val = MsgPackValue()
            val.type = 2
            val.int_val = Int64(Int8(read_byte()))
            return val
        if b == MSGPACK_INT16:
            var ptr = read_bytes(2)
            var val = MsgPackValue()
            val.type = 2
            val.int_val = Int64(Int16((UInt16(ptr[0]) << 8) | UInt16(ptr[1])))
            return val
        if b == MSGPACK_INT32:
            var ptr = read_bytes(4)
            var val = MsgPackValue()
            val.type = 2
            var result: UInt32 = (UInt32(ptr[0]) << 24) | (UInt32(ptr[1]) << 16) | (UInt32(ptr[2]) << 8) | UInt32(ptr[3])
            val.int_val = Int64(Int32(result))
            return val
        if b == MSGPACK_INT64:
            var ptr = read_bytes(8)
            var val = MsgPackValue()
            val.type = 2
            var result: UInt64 = 0
            for i in range(8):
                result = (result << 8) | UInt64(ptr[i])
            val.int_val = Int64(result)
            return val

        # str8/16/32
        if b == MSGPACK_STR8:
            var str_len = Int(read_byte())
            var ptr = read_bytes(str_len)
            var val = MsgPackValue()
            val.type = 4
            val.str_val = String(ptr, str_len)
            return val
        if b == MSGPACK_STR16:
            var ptr = read_bytes(2)
            var str_len = Int((UInt16(ptr[0]) << 8) | UInt16(ptr[1]))
            var data_ptr = read_bytes(str_len)
            var val = MsgPackValue()
            val.type = 4
            val.str_val = String(data_ptr, str_len)
            return val
        if b == MSGPACK_STR32:
            var ptr = read_bytes(4)
            var str_len = Int((UInt32(ptr[0]) << 24) | (UInt32(ptr[1]) << 16) | (UInt32(ptr[2]) << 8) | UInt32(ptr[3]))
            var data_ptr = read_bytes(str_len)
            var val = MsgPackValue()
            val.type = 4
            val.str_val = String(data_ptr, str_len)
            return val

        # array16/32
        if b == MSGPACK_ARRAY16:
            var ptr = read_bytes(2)
            var count = Int((UInt16(ptr[0]) << 8) | UInt16(ptr[1]))
            var val = MsgPackValue()
            val.type = 6
            val.array_values = List[MsgPackValue]()
            for _ in range(count):
                val.array_values.append(decode_value())
            return val
        if b == MSGPACK_ARRAY32:
            var ptr = read_bytes(4)
            var count = Int((UInt32(ptr[0]) << 24) | (UInt32(ptr[1]) << 16) | (UInt32(ptr[2]) << 8) | UInt32(ptr[3]))
            var val = MsgPackValue()
            val.type = 6
            val.array_values = List[MsgPackValue]()
            for _ in range(count):
                val.array_values.append(decode_value())
            return val

        # map16/32
        if b == MSGPACK_MAP16:
            var ptr = read_bytes(2)
            var count = Int((UInt16(ptr[0]) << 8) | UInt16(ptr[1]))
            var val = MsgPackValue()
            val.type = 5
            val.map_keys = List[String]()
            val.map_values = List[MsgPackValue]()
            for _ in range(count):
                var key = decode_value()
                var value = decode_value()
                val.map_keys.append(key.str_val)
                val.map_values.append(value)
            return val
        if b == MSGPACK_MAP32:
            var ptr = read_bytes(4)
            var count = Int((UInt32(ptr[0]) << 24) | (UInt32(ptr[1]) << 16) | (UInt32(ptr[2]) << 8) | UInt32(ptr[3]))
            var val = MsgPackValue()
            val.type = 5
            val.map_keys = List[String]()
            val.map_values = List[MsgPackValue]()
            for _ in range(count):
                var key = decode_value()
                var value = decode_value()
                val.map_keys.append(key.str_val)
                val.map_values.append(value)
            return val

        raise Error("MsgPack: unsupported format 0x" + hex(b))

    return decode_value()
