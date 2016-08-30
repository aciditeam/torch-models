-- Simple socket test script

local socket = require 'socket'
local osc = require 'osc'

local data = osc.pack('/some/url', 4)

local udp = assert(socket.udp())

local host = "localhost"
-- convert host name to ip address
local ip = assert(socket.dns.toip(host))

local port = 7402

assert(udp:sendto(data, ip, port))

udp:close()
