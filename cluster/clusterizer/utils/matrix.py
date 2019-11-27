from operator import itemgetter
from scipy.sparse import coo_matrix, vstack, hstack

# def replace_row(docspace, row, row_ndx):
#     if docspace.shape[0] == 1:
#         return row

#     if row_ndx == 0:
#         docspace = vstack([row, docspace[row_ndx+1:, :]], format='csr')
#     elif row_ndx == (docspace.shape[0] - 1):
#         docspace = vstack([docspace[:row_ndx, :], row], format='csr')
#     else:
#         docspace = vstack([docspace[:row_ndx, :],
#                        row,
#                        docspace[row_ndx+1:, :]], format='csr')
#     return docspace


def add_replace_rows(matrix, rows, format='csr'):
    "Replace rows in docspace. Gets rows list as (row_ndx, row)"
    return vstack(replace_row_helper(matrix, rows), format=format)


def set_columns_number(matrix, col_n, format='csr'):
    if matrix is None:
        return None
    m, n = matrix.shape
    if n < col_n:
        new_columns = coo_matrix((m, col_n - n))
        return hstack([matrix, new_columns], format=format)
    else:
        return matrix


def replace_row_helper(matrix, rows):
    rows.sort(key=itemgetter(0))
    if matrix is None:
        idxes = map(itemgetter(0), rows)
        assert idxes == range(len(idxes))
        for ndx, row in rows:
            yield row

    m, n = matrix.shape
    prev = 0
    # 1,4,5,6,7
    # 2:4 wasnt' used
    for ndx, row in rows:
        if prev < m and ndx > prev:
            yield matrix[prev:min(ndx, m), :]
        yield row
        prev = ndx + 1

    last_ndx = rows[-1][0]
    if last_ndx < (m - 1):
        yield matrix[last_ndx + 1:, :]


def remove_rows(matrix, row_ndxes, format='csr'):
    row_ndxes.sort()

    m, n = matrix.shape
    ranges = []
    prev = 0
    for i in row_ndxes:
        if i > prev:
            ranges.append((prev, i))
        prev = i + 1
    if row_ndxes[-1] < (m - 1):
        ranges.append((row_ndxes[-1] + 1, m))

    if ranges:
        matrix = vstack((matrix[b:e, :] for (b, e) in ranges), format=format)
    else:
        matrix = None

    return matrix


def append_rows(matrix, new_rows, format='csr'):
    # max_feat = max([r.shape[1] for r in new_rows])
    dim2 = [r.shape[1] for r in new_rows]
    if matrix is not None:
        dim2.append(matrix.shape[1])

    max_feat = max(dim2)
    new_rows = [set_columns_number(row, max_feat, format=format) for row in new_rows]
    matrix2 = vstack(new_rows, format=format)
    if matrix is not None:
        matrix = set_columns_number(matrix, matrix2.shape[1], format)
        return vstack([matrix, matrix2], format=format)
    else:
        return matrix2
