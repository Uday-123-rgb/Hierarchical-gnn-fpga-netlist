#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <math.h>

#define Filename "siso_bidirectional_16bit.edn"

FILE *fptr3;

#define MAX_TERMS 2000

typedef struct {
    int value;         // Bit values (0 or 1) where mask bit is 0
    int mask;          // 1 indicates a don't-care ('-') position
    uint64_t covered;  // Bitmask tracking which original minterms are covered
    int used;          // Flag to check if this term combined with another
} Implicant;

// Helper: Portable bit counter (Brian Kernighan's algorithm)
int count_ones(uint64_t n) {
    int count = 0;
    while (n) {
        n &= (n - 1);
        count++;
    }
    return count;
}

// Helper: Parses LUT strings like "4'h8", "16'h8000", or "0x8000"
void parse_lut_string(const char *lut_str, uint64_t *lut_mask, int *num_vars) {
    int width = 0;
    const char *hex_ptr = strchr(lut_str, 'h');
    
    if (hex_ptr) {
        width = atoi(lut_str);
        hex_ptr++; // Move past 'h'
    } else {
        if (strncmp(lut_str, "0x", 2) == 0 || strncmp(lut_str, "0X", 2) == 0) {
            hex_ptr = lut_str + 2;
        } else {
            hex_ptr = lut_str;
        }
        width = strlen(hex_ptr) * 4;
    }
    
    *num_vars = (int)(log2(width) + 0.5);
    if (*num_vars > 6) *num_vars = 6; // Cap at 6 variables

    *lut_mask = 0;
    while (*hex_ptr) {
        char c = *hex_ptr;
        int digit = 0;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
        else { hex_ptr++; continue; }
        
        *lut_mask = (*lut_mask << 4) | digit;
        hex_ptr++;
    }
}

// Helper: Formats and prints the minimized Boolean expression
void print_expression(Implicant *pis, int *chosen, int count, int num_vars) {
    if (count == 0) {
        printf("Output: 0\n");
        fprintf(fptr3,"Output: 0\n");
        return;
    }
    
    for (int i = 0; i < count; i++) {
        if (pis[chosen[i]].mask == ((1 << num_vars) - 1)) {
            printf("Output: 1\n");
            fprintf(fptr3,"Output: 1\n");
            return;
        }
    }

    printf("Output: ");
    for (int i = 0; i < count; i++) {
        Implicant pi = pis[chosen[i]];
        int printed_var = 0;
        
        for (int v = 0; v < num_vars; v++) {
            int bit_pos = num_vars - 1 - v;
            if (!((pi.mask >> bit_pos) & 1)) {
                char var_name = 'A' + v;
                int bit_val = (pi.value >> bit_pos) & 1;
                if (bit_val) {
                    printf("%c", var_name);
                    fprintf(fptr3,"%c", var_name);
                } else {
                    printf("%c'", var_name); // Apostrophe represents NOT
                    fprintf(fptr3,"%c'", var_name);
                }
                printed_var = 1;
            }
        }
        if (!printed_var) {
            printf("1");
            fprintf(fptr3,"1");

        }
        if (i < count - 1) {
            printf(" + ");
            fprintf(fptr3," + ");
        }
    }
    fprintf(fptr3,"\n");
    printf("\n");
}

// --- The Requested Modification Function ---
void decode_lut(const char *lut_str) {
    uint64_t lut_mask = 0;
    int num_vars = 0;
    
    parse_lut_string(lut_str, &lut_mask, &num_vars);

    printf("String: %s -> (%d vars, Mask: 0x%016llX)\n", lut_str, num_vars, (unsigned long long)lut_mask);

    if (lut_mask == 0) {
        printf("Output: 0\n\n");
        return;
    }

    // Step 1: Initialize generation 0 with minterms
    Implicant current[MAX_TERMS];
    int current_count = 0;
    uint64_t total_minterms = 1ULL << num_vars;

    for (uint64_t i = 0; i < total_minterms; i++) {
        if ((lut_mask >> i) & 1) {
            current[current_count].value = i;
            current[current_count].mask = 0;
            current[current_count].covered = (1ULL << i);
            current[current_count].used = 0;
            current_count++;
        }
    }

    Implicant prime_implicants[MAX_TERMS];
    int pi_count = 0;

    // Step 2: Combine terms iteratively (Quine-McCluskey Core)
    while (current_count > 0) {
        Implicant next_gen[MAX_TERMS];
        int next_count = 0;

        for (int i = 0; i < current_count; i++) current[i].used = 0;

        for (int i = 0; i < current_count; i++) {
            for (int j = i + 1; j < current_count; j++) {
                if (current[i].mask == current[j].mask) {
                    int diff = current[i].value ^ current[j].value;
                    if (count_ones(diff) == 1) {
                        current[i].used = 1;
                        current[j].used = 1;

                        int new_mask = current[i].mask | diff;
                        int new_value = current[i].value & ~diff;
                        uint64_t new_covered = current[i].covered | current[j].covered;

                        int dup = 0;
                        for (int k = 0; k < next_count; k++) {
                            if (next_gen[k].value == new_value && next_gen[k].mask == new_mask) {
                                next_gen[k].covered |= new_covered;
                                dup = 1;
                                break;
                            }
                        }
                        if (!dup) {
                            next_gen[next_count].value = new_value;
                            next_gen[next_count].mask = new_mask;
                            next_gen[next_count].covered = new_covered;
                            next_gen[next_count].used = 0;
                            next_count++;
                        }
                    }
                }
            }
        }

        for (int i = 0; i < current_count; i++) {
            if (!current[i].used) {
                int dup = 0;
                for (int k = 0; k < pi_count; k++) {
                    if (prime_implicants[k].value == current[i].value && prime_implicants[k].mask == current[i].mask) {
                        dup = 1; break;
                    }
                }
                if (!dup) {
                    prime_implicants[pi_count++] = current[i];
                }
            }
        }

        current_count = next_count;
        memcpy(current, next_gen, sizeof(Implicant) * next_count);
    }

    // Step 3: Prime Implicant Chart Solver
    int chosen_pis[MAX_TERMS];
    int chosen_count = 0;
    uint64_t remaining_minterms = lut_mask;

    // 3a. Find Essential Prime Implicants (EPIs)
    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = 0; i < (1 << num_vars); i++) {
            if ((remaining_minterms >> i) & 1) {
                int cover_count = 0;
                int last_pi_idx = -1;
                for (int j = 0; j < pi_count; j++) {
                    if ((prime_implicants[j].covered >> i) & 1) {
                        cover_count++;
                        last_pi_idx = j;
                    }
                }
                if (cover_count == 1) {
                    int already_chosen = 0;
                    for (int k = 0; k < chosen_count; k++) {
                        if (chosen_pis[k] == last_pi_idx) {
                            already_chosen = 1; break;
                        }
                    }
                    if (!already_chosen) {
                        chosen_pis[chosen_count++] = last_pi_idx;
                        remaining_minterms &= ~prime_implicants[last_pi_idx].covered;
                        changed = 1;
                    }
                }
            }
        }
    }

    // 3b. Greedy pass over remaining minterms
    while (remaining_minterms != 0) {
        int best_pi = -1;
        int max_covered = 0;
        for (int j = 0; j < pi_count; j++) {
            uint64_t intersection = prime_implicants[j].covered & remaining_minterms;
            int count = count_ones(intersection);
            if (count > max_covered) {
                max_covered = count;
                best_pi = j;
            }
        }
        if (best_pi == -1) break; 
        chosen_pis[chosen_count++] = best_pi;
        remaining_minterms &= ~prime_implicants[best_pi].covered;
    }

    // Step 4: Display Output
    printf("^\n");
    fprintf(fptr3,"^\n");
    print_expression(prime_implicants, chosen_pis, chosen_count, num_vars);
    printf("^\n");
    fprintf(fptr3,"^\n");
    //printf("\n"); // Clear trailing space for clean test blocks
}



//problem here
//not anymore XD

void string_finder1(char *str9,FILE *fptr1){
    char *str10 = malloc(32*sizeof(char));
    char *str11 = "cell";
    char d = fgetc(fptr1);
    if(d == EOF){
        free(str10);
        return;
    }
    if(d==' '|d=='('|d==')'){
        free(str10);
        string_finder1(str9,fptr1);
        return;
    }
    int i = 0;
    do{
        str10[i]=d;
        d=fgetc(fptr1);
        i++;
    }while(d!=' ' && !feof(fptr1) && !(d=='('||d==')') );
    str10[i]='\0';
    int cmp = strncmp(str9,str10,strlen(str9)+1);
    int cmp2 = strncmp(str10,str11,strlen(str11)+1);
    free(str10);
    if(!cmp2){
        return;
    }
    if (cmp){
        string_finder1(str9,fptr1);
        return;
        
    }

    //printf("%s\n",str10);
    d = fgetc(fptr1);
    
    char *str12 = malloc(32*sizeof(char));
    int k = 0;
    do{
        str12[k]=d;
        d=fgetc(fptr1);
        k++;
    }while(d!=' ' && !feof(fptr1) && !(d=='('||d==')') );
    str12[k]='\0';
    char e;
    e = fgetc(fptr1);
    e = fgetc(fptr1);
    e = fgetc(fptr1);
    e = fgetc(fptr1);
    e = fgetc(fptr1);
    e = fgetc(fptr1);
    e = fgetc(fptr1);
    e = fgetc(fptr1);
    e = fgetc(fptr1);
    e = fgetc(fptr1);
    e = fgetc(fptr1);
    e = fgetc(fptr1);
    int n =0;

    char *str13 = malloc(32*sizeof(char));
    do{
        str13[n]=e;
        e=fgetc(fptr1);
        n++;
    }while(e!=' ' && !feof(fptr1) && !(e=='('||e==')') );
    str13[n]='\0';
    printf("&\n");
    fprintf(fptr3,"&\n");
    printf("%s\n",str12);
    
    fprintf(fptr3,"%s\n",str12);
    printf("&\n");
    fprintf(fptr3,"&\n");
    printf("*\n");
    fprintf(fptr3,"*\n");
    printf("%s\n",str13);
    fprintf(fptr3,"%s\n",str13);
    printf("*\n");
    fprintf(fptr3,"*\n");
    
    
    string_finder1(str9,fptr1);
    return;

}

void string_finder(char *str6,FILE *fptr1){
    char *str7 = malloc(32*sizeof(char));
    char d = fgetc(fptr1);
    if(d == EOF){
        free(str7);
        return;
    }
    if(d==' '|d=='('|d==')'){
        free(str7);
        string_finder(str6,fptr1);
        return;
    }
    int i = 0;
    do{
        str7[i]=d;
        d=fgetc(fptr1);
        i++;
    }while(d!=' ' && !feof(fptr1) && !(d=='('||d==')') );
    str7[i]='\0';
    int cmp = strncmp(str6,str7,strlen(str6)+1);
    if (cmp){
        string_finder(str6,fptr1);
        return;
        
    }
    
    printf("#\n");
    fprintf(fptr3,"#\n");
    printf("%s\n",str7);
    fprintf(fptr3,"%s\n",str7);
    printf("#\n");
    fprintf(fptr3,"#\n");
    free(str7);

    return;

}

//issue in the ports function , will find tomorrow
void ports(FILE *fptr,char c){
    char *str4 = malloc(32*sizeof(char));
    //c = fgetc(fptr);
    if(c==' '|c=='('|c==')'){
        free(str4);
        c = fgetc(fptr);
        ports(fptr,c);
        return;
    }
    char str5[] = "cellref";
    int i = 0;
    do{
        str4[i]=c;
        c=fgetc(fptr);
        i++;
    }while(c!=' ' && !feof(fptr) && !(c=='('||c==')') );
    str4[i]='\0';
    int cmp=strncmp(str4,str5,8);
    if(cmp){
        free(str4);
        ports(fptr,c);
        return;
    }
    //printf("%s\n",str4);
    free(str4);
    c = fgetc(fptr);
    int j =0;
    char *str8 = malloc(32*sizeof(char));
    do{
        
        str8[j]=c;
        c=fgetc(fptr);
        j++;

    }while(c!=' ' && !feof(fptr) && !(c=='('||c==')') );
    str8[j]='\0';
    char *str99 = "LUT";
    int cmp1 = strncmp(str8,str99,3);
    if(!cmp1){
        char str98[100];
        fgets(str98,100,fptr);
        fgets(str98,100,fptr);
        char *myPtr = strchr(str98, '"');
            if (myPtr != NULL) {
                char lut[32];
                int r = 0;
                do{
                    if(myPtr[r+1]!='\"'){
                        lut[r]=myPtr[r+1];
                    }
                        r++;
                }while(myPtr[r]!='\"');
                lut[r-1]='\0';
                decode_lut(lut);
            }
    }
    char *str9 = "port";
    FILE *fptr1;
    fptr1 = fopen(Filename,"r");
    string_finder(str8,fptr1);
    string_finder1(str9,fptr1);
    fclose(fptr1);
    fptr1 = NULL;
    free(str8);
    
    return;
}

void cellref(FILE *fptr,char c){
    char *str3 = malloc(32*sizeof(char));
    int i=0;
    c = fgetc(fptr);
    do{
        str3[i]=c;
        c=fgetc(fptr);
        i++;
    }while(c!=' ' && !feof(fptr) && !(c=='('||c==')') );
    str3[i]='\0';
    char *str14 = "(rename";
    int cmp3 = strncmp(str3,str14,7);
    
    if(!cmp3){
        int y = 0;
        c = fgetc(fptr);
        do{
        str3[y]=c;
        c=fgetc(fptr);
        y++;
    }while(c!=' ' && !feof(fptr) && !(c=='('||c==')') );
    str3[y]='\0';
    }
    printf("$\n");
    fprintf(fptr3,"$\n");
    //printf("\n");
    printf("%s\n",str3);
    //fprintf(fptr3,"\n");
    fprintf(fptr3,"%s\n",str3);
    printf("$\n");
    fprintf(fptr3,"$\n");
    free(str3);
    ports(fptr,c);
    return;

}

void write_num(FILE *fptr,char c){
    char *str1 = malloc(32*sizeof(char));
    int i=0;
    do{
        str1[i]=c;
        c=fgetc(fptr);
        i++;
        
    }while(c!=' ' && !feof(fptr) && !(c=='('||c==')') && !isalpha(c));
    str1[i]='\0';
    printf("%s\n",str1);
    fprintf(fptr3,"%s\n",str1);
    free(str1);

    str1 = NULL;
    return;
}
void instance(char c,FILE *fptr){
    char *str1 = malloc(32*sizeof(char));
    int i=0;
    do{
        str1[i]=c;
        c=fgetc(fptr);
        i++;
    }while(c!=' ' && !feof(fptr) && !(c=='('||c==')') );
    str1[i]='\0';
    char str2[]="instance";
    int cmp = strncmp(str1,str2,10);
    if (!cmp){
        cellref(fptr,c);
        //printf("%s\n",str1);
    }
    
    free(str1);

    str1 = NULL;
    return;
}

void lexer(FILE *fptr){
    char c = fgetc(fptr);
    //if(c=='('||c==')')printf("%c\n",c);
     if(isalpha(c)){
        instance(c,fptr);
    }
    /*else if(isdigit(c)){
        write_num(fptr,c);
    }*/
    if (!feof(fptr))lexer(fptr);
    return;

}
void basic_ports(FILE *fptr2){
    char *str15 = malloc(32*sizeof(char));
    int i=0;
    char *str16 = "work";
    char c = fgetc(fptr2);
    if (c == EOF) {
        free(str15);
        return;
    }
    if(c==' ' || c== '(' || c == ')'){
        free(str15);
        basic_ports(fptr2);
        return;
    }
    do{
        str15[i]=c;
        c=fgetc(fptr2);
        i++;
    }while(c!=' ' && !feof(fptr2) && !(c=='('||c==')') );
    str15[i]='\0';
    int cmp4 = strncmp(str15,str16,4);
    if(!cmp4) {
        //printf("%s",str15);
        return;
    }
    free(str15);
    basic_ports(fptr2);
    return;

}
void array_expansion(char *str22,char *str27,char *str23){
    int i = 0 ;
    int j = 0 ;
    int k,l;

    char c = str22[i];
    char d = str22[j];
    char *str26 = str22;
    while(!isdigit(d)){
        j++;
        d= str22[j];
    }
    k = d - '0';
    d=str22[j+2];
    l = d - '0';
    while(isalpha(c) | (c=='_')){
        str27[i] = c;
        i++;
        c = str26[i];
    }
    str27[i]='\0';
    int n =(k-l+1);
    for(int m=0;m<n;m++){
        printf("$\n");
        fprintf(fptr3,"$\n");
        printf("%s_%d_\n",str27,m);
        fprintf(fptr3,"%s_%d_\n",str27,m);
        printf("$");
        fprintf(fptr3,"$");
        printf("\n#\n");
        fprintf(fptr3,"\n#\n");
        printf("%s\n",str23);
        fprintf(fptr3,"%s\n",str23);
        printf("#\n");
        fprintf(fptr3,"#\n");
    }
    return;
}

//:( problem here , after lunch

void printing_basic_ports(FILE *fptr2){
    char *str17 ="port";
    char *str18 = malloc(32*sizeof(char));
    char *str19 = "contents";
    char c= fgetc(fptr2);
    if (feof(fptr2)) {
        free(str18);
        return;
    }
    if(c==' ' || c== '(' || c == ')'){
        free(str18);
        printing_basic_ports(fptr2);
        return;
    }
    int i = 0;
    do{
        str18[i]=c;
        c=fgetc(fptr2);
        i++;
    }while(c!=' ' && !feof(fptr2) && !(c=='('||c==')') );
    str18[i]='\0';
    int cmp5 = strncmp(str17,str18,4);
    int cmp6 = strncmp(str18,str19,8);
    if(!cmp6) return;
    if(!cmp5){
        char d = fgetc(fptr2);
        char *str20 = malloc(32*sizeof(char));
        char *str21 = "(array";
        int q = 0;
        do{
            str20[q]=d;
            d=fgetc(fptr2);
            q++;
        }while(d!=' ' && !feof(fptr2) && !(d=='('||d==')') );
        str20[q]='\0';
        char *str22 = malloc(32*sizeof(char));
        char *str23 = malloc(32*sizeof(char));
        char *str24 = malloc(32*sizeof(char));
        
        int cmp7 = strncmp(str20,str21,7);
        if (!cmp7){
            int w =0;
            
            d=fgetc(fptr2);
            d=fgetc(fptr2);
            d=fgetc(fptr2);
            d=fgetc(fptr2);
            d=fgetc(fptr2);
            d=fgetc(fptr2);
            d=fgetc(fptr2);
            d=fgetc(fptr2);
            d=fgetc(fptr2);

            do{
                str20[w]=d;
                d=fgetc(fptr2);
                w++;
            }while(d!=' ' && !feof(fptr2) && !(d=='('||d==')') );
            str20[w]='\0';
            int p =-1;
            d = fgetc(fptr2);
            do{
                str22[p]=d;
                d=fgetc(fptr2);
                p++;
            }while(d!=' ' && !feof(fptr2) && !(d=='('||d==')') && d!='\"');
            str22[p]='\0';
            d = fgetc(fptr2);
            while(d!='n'){
                d=fgetc(fptr2);
            }
            d=fgetc(fptr2);
            d=fgetc(fptr2);
            p=0;
            do{
                str23[p]=d;
                d=fgetc(fptr2);
                p++;
            }while(d!=' ' && !feof(fptr2) && !(d=='('||d==')') );
            str23[p]='\0';
            
        }
        if(cmp7){
        d = fgetc(fptr2);
        while(d!='n'){
            d=fgetc(fptr2);
        }
        d=fgetc(fptr2);
        d=fgetc(fptr2);
        int p=0;
        do{
            str24[p]=d;
            d=fgetc(fptr2);
            p++;
        }while(d!=' ' && !feof(fptr2) && !(d=='('||d==')') );
        str24[p]='\0';
        printf("$\n");
        fprintf(fptr3,"$\n");
        printf("%s\n",str20);
        fprintf(fptr3,"%s\n",str20);
        printf("$");
        fprintf(fptr3,"$");

        if(cmp7) {
            printf("\n#");
            fprintf(fptr3,"\n#");
            printf("\n");
            printf("%s\n",str24);
            fprintf(fptr3,"\n");
            fprintf(fptr3,"%s\n",str24);
            printf("#\n");
            fprintf(fptr3,"#\n");
        }}
        if (!cmp7) {
            char *str27 = malloc(32*sizeof(char));
            array_expansion(str22,str27,str23);

            
            
        }
        
    }
    free (str18);
    printing_basic_ports(fptr2);

    return;

}

int main() {
    fptr3 = fopen("parsed.text","w");
    FILE *fptr2;
    fptr2 = fopen(Filename,"r");
    /*printf("$\n");
    fprintf(fptr3,"$\n");
    printf("black_box_ports\n");
    fprintf(fptr3,"black_box_ports\n");
    printf("$\n");
    fprintf(fptr3,"$\n");*/
    basic_ports(fptr2);
    printing_basic_ports(fptr2);
    fclose(fptr2);
    FILE *fptr;
    //printf("\n");
    //fprintf(fptr3,"\n");
    fptr = fopen(Filename,"r");
    printf("$\n");
    //fprintf(fptr3,"$\n");
    lexer(fptr);
    fclose(fptr);
    fclose(fptr3);
    return 0;
   
}