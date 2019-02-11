# get the testers
TESTING_FILES=$(cd bin && ls testing_*)

# run each tester
for i in $TESTING_FILES; do
    printf "==== TESTING %s ====\n" "$i"
    ./bin/$i
    printf "\n"
done