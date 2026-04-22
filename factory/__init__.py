"""Bug-finding factory.

Independent daemons cooperate via a shared SQLite DB to continuously:
  1. generate PTX programs
  2. diff OpenPTXas vs ptxas on the GPU
  3. minimize divergent programs
  4. cluster by op-family signature
  5. decide whose output matches spec
  6. auto-emit PSIRT-format dossiers for ptxas bugs
"""
