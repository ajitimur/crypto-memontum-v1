"""The simulation core.

Pure by construction: nothing in this package opens a socket, touches the
filesystem, or reads the clock. A result is a function of the bars and the
parameters it was handed, which is what makes a run reproducible from a commit
hash and a config file. A test asserts this structurally.
"""
