from wisent.benchmarks.coding.safe_docker.core.runtime import DockerSandboxExecutor
from wisent.benchmarks.coding.safe_docker.recipes import RECIPE_REGISTRY

files = {
    "Solution.java": "public class Solution{public static int add(int a,int b){return a+b;}}",
    "MainTest.java": "public class MainTest{public static void main(String[]a){"
                     "if(Solution.add(1,2)!=3)throw new RuntimeException(\"fail\");"
                     "System.out.println(\"OK\");}}",
}
job = RECIPE_REGISTRY["java"].make_job(files, java_main="MainTest", time_limit_s=6)
res = DockerSandboxExecutor(image="coding/sandbox:polyglot-1.0").run(files, job)
print(res)
