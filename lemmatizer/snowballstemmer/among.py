# Copyright (c) 2001, Dr Martin Porter
# Copyright (c) 2004,2005, Richard Boulton
# Copyright (c) 2013, Yoshiki Shibukawa
# Copyright (c) 2006,2007,2009,2010,2011,2014-2019, Olly Betts
# All rights reserved.


class Among(object):
    def __init__(self, s, substring_i, result, method=None):
        """
        @ivar s search string
        @ivar substring index to longest matching substring
        @ivar result of the lookup
        @ivar method method to use if substring matches
        """
        self.s = s
        self.substring_i = substring_i
        self.result = result
        self.method = method
