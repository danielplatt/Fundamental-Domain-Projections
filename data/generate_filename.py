

def generate_filename(projection_type: str, is_permuted: bool):
    file_name = 'cicy_'
    if projection_type == 'dirichlet':
        file_name += 'dirichlet_from_'
    elif projection_type == 'combinatorial':
        file_name += 'combinatorial_from_'
    if is_permuted:
        file_name += 'permuted.pckl'
    else:
        file_name += 'original.pckl'
    return file_name


if __name__ == '__main__':
    print(generate_filename('dirichlet', True))
