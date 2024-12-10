if __name__ == '__main__':
    save_score = './score.trl.txt'
    with open(save_score, "a") as fh:
        fh.write("{}\t{}\t{}\n".format(str(1),str(2),str(3)))  # [-3.22324123245] 去除[ ]然后转为str存储
        print("Scores saved to {}".format(save_score))
    print('End of Program.')