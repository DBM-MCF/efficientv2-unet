from subprocess import check_call, check_output, STDOUT

# TODO implement these tests
'''
Notes: maybe good if I just run some train and prediction commands 
with random images.
making sure everything works as intended.
'''


if __name__ == '__main__':
    pass
    '''
    #cli.get_arg_parser().parse_args()
    cmd = 'ev2unet --help'
    try:
        #cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
        cmd_call = check_call(cmd, shell=True)
        print('------- cmdoutput -------')
        print(cmd_call)
    except Exception as e:
        print(e)
        raise ValueError(e)
    '''
