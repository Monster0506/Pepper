? 10

%% String Library Tests
LET test_str: string = "Hello, World!"
LET upper_str: string = upper FROM string test_str _
SHOW(upper_str)
LET lower_str: string = lower FROM string test_str _
SHOW(lower_str)
LET str_len: int = len FROM string test_str _
SHOW(str_len)
LET trimmed: string = trim FROM string "  Hello  " _
SHOW(trimmed)
LET replaced: string = replace FROM string test_str ("o", "0")
SHOW(replaced)
LET split_list: list = split FROM string "a,b,c" (",")
SHOW(split_list)
LET joined: string = join FROM string "," (["x", "y", "z"])
SHOW(joined)
LET has_world: bool = contains FROM string test_str ("World")
SHOW(has_world)
LET starts_hello: bool = starts_with FROM string test_str ("Hello")
SHOW(starts_hello)
LET ends_exclaim: bool = ends_with FROM string test_str ("!")
SHOW(ends_exclaim)
LET repeated: string = repeat FROM string "Ha" (3)
SHOW(repeated)
LET reversed_str: string = reverse FROM string test_str _
SHOW(reversed_str)
LET substring: string = substr FROM string test_str (0, 5)
SHOW(substring)
LET padded_left: string = pad_left FROM string "42" (5, "0")
SHOW(padded_left)
LET padded_right: string = pad_right FROM string "42" (5, "0")
SHOW(padded_right)
LET capitalized: string = capitalize FROM string "hello world" _
SHOW(capitalized)
LET titled: string = title FROM string "hello world" _
SHOW(titled)

%% Math Library Tests
LET num: float = 16.0
LET sqrt_num: float = sqrt FROM math num _
SHOW(sqrt_num)
LET powered: float = pow FROM math 2.0 (3)
SHOW(powered)
LET abs_num: float = abs FROM math -42.5 _
SHOW(abs_num)
LET floored: int = floor FROM math 3.7 _
SHOW(floored)
LET ceiled: int = ceil FROM math 3.2 _
SHOW(ceiled)
LET rounded: float = round FROM math 3.14159 (2)
SHOW(rounded)
LET min_val: int = min FROM math [1, 2, 3, 4, 5] _
SHOW(min_val)
LET max_val: int = max FROM math [1, 2, 3, 4, 5] _
SHOW(max_val)

%% Continue with rest of math functions...
LET sin_val: float = sin FROM math 0.0 _
SHOW(sin_val)
LET cos_val: float = cos FROM math 0.0 _
SHOW(cos_val)
LET tan_val: float = tan FROM math 0.0 _
SHOW(tan_val)
LET asin_val: float = asin FROM math 0.0 _
SHOW(asin_val)
LET acos_val: float = acos FROM math 1.0 _
SHOW(acos_val)
LET atan_val: float = atan FROM math 0.0 _
SHOW(atan_val)
LET log_val: float = log FROM math 2.718281828459045 _
SHOW(log_val)
LET log10_val: float = log10 FROM math 100.0 _
SHOW(log10_val)
LET exp_val: float = exp FROM math 1.0 _
SHOW(exp_val)
LET pi_val: float = pi FROM math _ _
SHOW(pi_val)
LET e_val: float = e FROM math _ _
SHOW(e_val)
LET fact_val: int = factorial FROM math 5 _
SHOW(fact_val)
LET gcd_val: int = gcd FROM math 48 (18)
SHOW(gcd_val)
LET lcm_val: int = lcm FROM math 48 (18)
SHOW(lcm_val)
LET is_prime_val: bool = is_prime FROM math 17 _
SHOW(is_prime_val)
LET degrees_val: float = degrees FROM math 3.1415926535897932384626433832795 _
SHOW(degrees_val)
LET radians_val: float = radians FROM math 180.0 _
SHOW(radians_val)

%% Random Library Tests (using seed for reproducibility)
LET seed: int = 42
LET rand_float: float = rnd FROM random seed _
SHOW(rand_float)
LET rand_int: int = rnd FROM random seed (10)
SHOW(rand_int)
LET rand_range: int = rnd FROM random seed (1, 6)
SHOW(rand_range)
LET chosen: int = choice FROM random [1, 2, 3, 4, 5] _
SHOW(chosen)
LET shuffled: list = shuffle FROM random [1, 2, 3, 4, 5] _
SHOW(shuffled)
LET sampled: list = sample FROM random [1, 2, 3, 4, 5] (3)
SHOW(sampled)

%% Type Library Tests
LET is_int_val: bool = is_int FROM type 42 _
SHOW(is_int_val)
LET is_float_val: bool = is_float FROM type 3.14 _
SHOW(is_float_val)
LET is_string_val: bool = is_string FROM type "hello" _
SHOW(is_string_val)
LET is_list_val: bool = is_list FROM type [1, 2, 3] _
SHOW(is_list_val)
LET is_bool_val: bool = is_bool FROM type true _
SHOW(is_bool_val)
LET type_val: string = get FROM type 42 _
SHOW(type_val)
LET int_val: int = to_int FROM type "42" _
SHOW(int_val)
LET float_val: float = to_float FROM type "3.14" _
SHOW(float_val)
LET string_val: string = to_string FROM type 42 _
SHOW(string_val)
LET bool_val: bool = to_bool FROM type "true" _
SHOW(bool_val)

%% List Library Tests
LET test_list: list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
LET sorted_list: list = sort FROM list test_list _
SHOW(sorted_list)
LET reverse_sorted: list = sort FROM list test_list (true)
SHOW(reverse_sorted)
LET reversed_list: list = reverse FROM list test_list _
SHOW(reversed_list)


%% LET mapped_list: list = map FROM list [1, 2, 3] (x => x 2 *)
%% SHOW(mapped_list)
%% LET filtered_list: list = filter FROM list [1, 2, 3, 4, 5] (x => x > 2)
%% SHOW(filtered_list)
%% LET reduced_val: int = reduce FROM list [1, 2, 3, 4, 5] ((x, y) => x + y)
%% SHOW(reduced_val)


LET found_idx: int = find FROM list test_list (5)
SHOW(found_idx)
LET count_val: int = count FROM list test_list (5)
SHOW(count_val)
LET sum_val: int = sum FROM list test_list _
SHOW(sum_val)
LET avg_val: float = avg FROM list test_list _
SHOW(avg_val)
LET zipped: list = zip FROM list [1, 2, 3] ([4, 5, 6])
SHOW(zipped)
LET enumerated: list = enumerate FROM list ["a", "b", "c"] _
SHOW(enumerated)
LET sliced: list = slice FROM list test_list (2, 5)
SHOW(sliced)
LET concatenated: list = concat FROM list [1, 2, 3] ([4, 5, 6])
SHOW(concatenated)
LET unique_list: list = unique FROM list test_list _
SHOW(unique_list)

%% Time Library Tests
LET current_time: float = now FROM time _ _
SHOW(current_time)
LET formatted_time: string = format FROM time 1704067200 ("%Y-%m-%d")
SHOW(formatted_time)
LET parsed_time: float = parse FROM time "2024-01-01" ("%Y-%m-%d")
SHOW(parsed_time)

%% File Library Tests
LET write_success: bool = write FROM file "test.txt" ("Hello, World!")
SHOW(write_success)
LET file_content: string = read FROM file "test.txt" _
SHOW(file_content)
LET file_exists: bool = exists FROM file "test.txt" _
SHOW(file_exists)
LET file_size: int = size FROM file "test.txt" _
SHOW(file_size)
LET rename_success: bool = rename FROM file "test.txt" ("test2.txt")
SHOW(rename_success)
LET new_exists: bool = exists FROM file "test2.txt" _
SHOW(new_exists)
LET delete_success: bool = delete FROM file "test2.txt" _
SHOW(delete_success)

%% System Library Tests
LET cmd_args: list = args FROM system _ _
SHOW(cmd_args)
LET path_var: string = env FROM system "PATH" _
SHOW(path_var)
LET cmd_output: string = exec FROM system "echo Hello from Pepper" _
SHOW(cmd_output)
