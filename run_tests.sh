# [32, 32, 512, 64]
# [16, 32, 1024, 64]
# [8, 32, 2048, 64]
# [4, 32, 4096, 64]
# [2, 32, 8192, 64]
# [1, 32, 16384, 64]

python3 python/test/test_forward_pass.py 32 32 512   64 > ./out/results/forward_pass_1_results.txt
python3 python/test/test_forward_pass.py 16 32 1024  64 > ./out/results/forward_pass_2_results.txt
python3 python/test/test_forward_pass.py 8  32 2048  64 > ./out/results/forward_pass_3_results.txt
python3 python/test/test_forward_pass.py 4  32 4096  64 > ./out/results/forward_pass_4_results.txt
python3 python/test/test_forward_pass.py 2  32 8192  64 > ./out/results/forward_pass_5_results.txt
python3 python/test/test_forward_pass.py 1  32 16384 64 > ./out/results/forward_pass_6_results.txt

python3 python/test/test_backward_pass.py 32 32 512   64 > ./out/results/backward_pass_1_results.txt
python3 python/test/test_backward_pass.py 16 32 1024  64 > ./out/results/backward_pass_2_results.txt
python3 python/test/test_backward_pass.py 8  32 2048  64 > ./out/results/backward_pass_3_results.txt
python3 python/test/test_backward_pass.py 4  32 4096  64 > ./out/results/backward_pass_4_results.txt
python3 python/test/test_backward_pass.py 2  32 8192  64 > ./out/results/backward_pass_5_results.txt
python3 python/test/test_backward_pass.py 1  32 16384 64 > ./out/results/backward_pass_6_results.txt