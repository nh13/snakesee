# Module containing reusable processing rules

rule preprocess:
    input: "input/{sample}.txt"
    output: "output/preprocessed/{sample}.txt"
    shell: "cat {input} | tr 'a-z' 'A-Z' > {output}"

rule analyze:
    input: "output/preprocessed/{sample}.txt"
    output: "output/analyzed/{sample}.txt"
    shell: "wc -c {input} > {output}"
